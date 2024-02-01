# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import base64
import copy
import json
import multiprocessing
multiprocessing.set_start_method('fork')

import os
import time
from collections import defaultdict
from multiprocessing import Manager, Process
from queue import Empty
import pandas as pd

import bottle
import cv2
import imageio.v2 as imageio

manager = Manager()
SAMPLE_QUEUE = manager.Queue()
OUTPUT_QUEUE = manager.Queue()
PENDING_LABELS = manager.dict()
LABEL_COUNTS = manager.dict()
TIMEOUT_THRESHOLD = 600  # in seconds
EXPECTED_ITERATION_TIME = 5  # in seconds
NUM_REQUEUE_TIMES = 3


def record_sample() -> str:
    bottle.response.content_type = "application/json"
    data = bottle.request.json

    # From https://stackoverflow.com/questions/31405812/how-to-get-client-ip-address-using-python-bottle-framework
    client_ip = bottle.request.environ.get(
        "HTTP_X_FORWARDED_FOR"
    ) or bottle.request.environ.get("REMOTE_ADDR")

    sample_idx = data["sample_idx"]

    if sample_idx in PENDING_LABELS:
        log_row = {
            "client_ip": client_ip,
            "email": data["email"],
            "out_time": str(data["time"]),
            "in_time": time.ctime(),
            "sample_idx": data["sample_idx"],
            "user_label": data["user_label"],
            "latitude": data["latitude"],
            "longitude": data["longitude"],
            "confidence": data["confidence"],
            "species": data["species"],
            "comments": data["comments"],
            "labeledby": data["labeledby"] + [data["email"]],
        }
        OUTPUT_QUEUE.put(log_row)
        del PENDING_LABELS[sample_idx]

        bottle.response.status = 200
        return json.dumps(data)
    else:
        # This can happen, for example, if someone loads the labeling page, then just
        # leaves it open for a few days between server restarts, then submits a label
        print("Received a label that we didn't actually ask for... ignoring")
        bottle.response.status = 200
        return json.dumps(data)


def get_sample() -> str:
    bottle.response.content_type = "application/json"
    data = bottle.request.json

    email = data["email"]

    # Get a sample to label that the current IP has not already labeled
    num_todo_samples = SAMPLE_QUEUE.qsize()
    if num_todo_samples == 0:
        print("No more samples to label in SAMPLE_QUEUE")
        bottle.response.status = 200
        return json.dumps({"error": "No more samples to label."})

    inputs = SAMPLE_QUEUE.get()
    i = 0
    while email in inputs["labeledby"] and i < num_todo_samples:
        i += 1
        SAMPLE_QUEUE.put(inputs)
        inputs = SAMPLE_QUEUE.get()

    if i >= num_todo_samples:
        SAMPLE_QUEUE.put(
            inputs
        )  # return the current sample back to the queue so that someone else can label it
        print(f"No more samples to label for {email}")
        bottle.response.status = 200
        return json.dumps({"error": "No more samples to label."})

    sample_idx = inputs["sample_idx"]
    fn = inputs["fn"]

    img = imageio.imread(fn)
    img_str = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[
        1
    ].tobytes()
    img_str = base64.b64encode(img_str).decode("utf-8")

    data["time"] = time.ctime()
    data["sample_idx"] = sample_idx
    data["img"] = img_str
    data["date"] = inputs["date"]
    data["latitude"] = inputs["latitude"]
    data["longitude"] = inputs["longitude"]
    data["labeledby"] = inputs["labeledby"]

    PENDING_LABELS[sample_idx] = (time.time(), inputs)

    bottle.response.status = 200
    return json.dumps(data)


def _data_loader_process(*args) -> None:
    """This is a wrapper that lets us gracefully shutdown when Ctrl+C is pressed.

    Note, we should write out anything remaining in `OUTPUT_QUEUE` to file here.
    """
    try:
        __data_loader_process(*args)
    except KeyboardInterrupt:
        print(
            f"[{multiprocessing.current_process().name}] ... Ctrl+C pressed, terminating ..."
        )


def __data_loader_process(
    output_fn_by_idx: str,
    locations_by_idx,
    fn_by_idx,
    date_by_idx,
    sampling_mode,
) -> None:
    """ """

    idxs = list(locations_by_idx.keys())

    # Add all of the samples to the queue to be labeled
    for idx in idxs:
        SAMPLE_QUEUE.put(
            {
                "sample_idx": idx,
                "fn": fn_by_idx[idx],
                "date": date_by_idx[idx],
                "latitude": locations_by_idx[idx][0],
                "longitude": locations_by_idx[idx][1],
                "labeledby": [],
            }
        )

    loop_tic = (
        time.time() - EXPECTED_ITERATION_TIME + 1
    )  # subtract some time on the first iteration
    while True:
        # ------------------------------------------------------------------------------
        # Check to see how long the previous loop took, sleep until the total time taken
        # is roughly `EXPECTED_ITERATION_TIME` seconds. We do this at the beginning instead
        # of at the end so we can use `continue` throughout the loop without skipping this.
        time_taken = time.time() - loop_tic
        sleep_time = EXPECTED_ITERATION_TIME - time_taken
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(f"WARNING! The monitor loop took {time_taken} seconds")
        loop_tic = time.time()

        # ------------------------------------------------------------------------------
        # Constantly monitor the PENDING_LABELS dictionary for items that have been sent
        # out for labeling, but we haven't gotten an answer for in more than
        # TIMEOUT_THRESHOLD seconds. If this happens, requeue that item.
        try:
            del_list = []
            for sample_idx, (out_time, inputs) in PENDING_LABELS.items():
                if time.time() - out_time > TIMEOUT_THRESHOLD:
                    print(f"Time out for sample {sample_idx}")
                    SAMPLE_QUEUE.put(inputs)
                    del_list.append(sample_idx)
            for sample_idx in del_list:
                del PENDING_LABELS[sample_idx]
        except Exception as e:
            print(f"Exception in PENDING_LABELS check: {e}")

        # ------------------------------------------------------------------------------
        # Check to see if we have finished a batch of labeling
        num_todo_samples = SAMPLE_QUEUE.qsize()
        num_finished_samples = OUTPUT_QUEUE.qsize()

        print(
            f"There are currently {num_finished_samples} samples that have been labeled, and {num_todo_samples} samples in the queue to be labeled"
        )

        # ------------------------------------------------------------------------------
        # Get labels from queue and record them
        rows = []
        while not OUTPUT_QUEUE.empty():
            try:
                row = copy.copy(OUTPUT_QUEUE.get(False))
                rows.append(row)  # record the row as to be written to file

                sample_idx = row["sample_idx"]
                num_times_labeled = len(row["labeledby"])
                if (num_times_labeled) < NUM_REQUEUE_TIMES:
                    print(
                        f"Sample {sample_idx} has been labeled {num_times_labeled}, requeing"
                    )
                    SAMPLE_QUEUE.put(
                        {
                            "sample_idx": sample_idx,
                            "fn": fn_by_idx[sample_idx],
                            "date": date_by_idx[sample_idx],
                            "latitude": locations_by_idx[sample_idx][0],
                            "longitude": locations_by_idx[sample_idx][1],
                            "labeledby": row["labeledby"],
                        }
                    )
                else:
                    print(
                        f"Sample {sample_idx} has been labeled {num_times_labeled+1} and is finished"
                    )

            except Empty:
                pass

        for row in rows:
            with open(output_fn_by_idx[row['sample_idx']], "a") as f:
                f.write(
                    f"{row['sample_idx']},{row['latitude']},{row['longitude']},{row['email']},{row['client_ip']},{row['out_time']},{row['in_time']},{row['user_label']},{row['confidence']},{row['species']},\"{row['comments']}\"\n"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
        default=False,
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host to bind to",
        default="0.0.0.0",
    )
    parser.add_argument(
        "--remote_host",
        type=str,
        required=True,
        help="Hostname for the labeling webpage to connect to",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to listen on",
        default=8080,
    )
    parser.add_argument(
        "--input_dirs",
        required=True,
        nargs="+",
        type=str,
        help="Root directory of run",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Flag to enable overwriting of `output_fn`",
    )
    parser.add_argument(
        "--sampling_mode",
        choices=["random", "active"],
        default="random",
        type=str,
        help="Root directory of run",
    )
    args = parser.parse_args()

    for input_dir in args.input_dirs:
        assert os.path.exists(input_dir), "The input file does not exist"
        for fn in ["inputs.csv", "metadata.json"]:
            assert os.path.exists(os.path.join(input_dir, fn))

    for input_dir in args.input_dirs:
        output_fn = os.path.join(input_dir, "labels.csv")
        if not os.path.exists(output_fn) or args.overwrite:
            with open(output_fn, "w") as f:
                f.write(
                    "sample_idx,latitude,longitude,email,client_ip,out_time,in_time,user_label,confidence,species,comments\n"
                )

    location_by_idx = dict()
    fn_by_idx = dict()
    date_by_idx = dict()
    output_fn_by_idx = dict()

    for input_dir in args.input_dirs:
        df = pd.read_csv(os.path.join(input_dir, "inputs.csv"))
        metadata = json.load(open(os.path.join(input_dir, "metadata.json"), "r"))
        for i in range(df.shape[0]):
            row = df.iloc[i]
            idx, lat, lon, fn = row["idx"], row["lat"], row["lon"], row["fn"]
            if fn.startswith("labeling-tool/"):
                fn = fn.replace("labeling-tool/", "")
            idx = int(idx)
            idx = f"{metadata['dar']}-{metadata['date']}-{metadata['catid']}-{idx}"
            lat = float(lat)
            lon = float(lon)
            location_by_idx[idx] = (lat, lon)
            date_by_idx[idx] = metadata["date"]
            fn_by_idx[idx] = fn
            output_fn_by_idx[idx] = os.path.join(input_dir, "labels.csv")

    # Start the monitoring / sampling loop
    p1 = Process(
        target=_data_loader_process,
        args=(output_fn_by_idx, location_by_idx, fn_by_idx, date_by_idx, args.sampling_mode),
    )
    p1.start()

    # Setup the bottle server
    app = bottle.Bottle()

    app.route("/recordSample", method="POST", callback=record_sample)
    app.route("/getSample", method="POST", callback=get_sample)

    app.route(
        "/",
        method="GET",
        callback=lambda: bottle.template(
            "index.html", host=args.remote_host, port=args.port
        ),
    )
    app.route("/favicon.ico", method="GET", callback=lambda: None)
    app.route(
        "/<filepath:re:.*>",
        method="GET",
        callback=lambda filepath: bottle.static_file(filepath, root=""),
    )

    bottle_server_kwargs = {
        "host": args.host,
        "port": args.port,
        "debug": args.verbose,
        "server": "tornado",
        "reloader": False,
    }
    app.run(**bottle_server_kwargs)


if __name__ == "__main__":
    main()
