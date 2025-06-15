import os
import time
import mlflow

from argparse import ArgumentParser

exp_id = mlflow.create_experiment("Demo")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Demo")

print("Tracking URI:", mlflow.get_tracking_uri())
print("Experiment ID:", exp_id)


def eval(p1, p2):
    output_metric = p1**2 + p2**2
    return output_metric


def main(inp1, inp2):
    with mlflow.start_run(run_name="Demo") as run:
        print("Run ID:", run.info.run_id)
        print("Artifact URI:", mlflow.get_artifact_uri())

        mlflow.set_tag("version", "1.0.0")
        mlflow.log_param("param1", inp1)
        mlflow.log_param("param2", inp2)

        metric = eval(inp1, inp2)
        mlflow.log_metric("Eval_metric", metric)

        os.makedirs("dummy", exist_ok=True)
        with open("dummy/example.txt", "wt") as f:
            f.write(time.asctime())

        mlflow.log_artifacts("dummy")
    mlflow.end_run()


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--param1", "-p1", type=int, default=5)
    args.add_argument("--param2", "-p2", type=int, default=10)
    parsed_args = args.parse_args()

    main(parsed_args.param1, parsed_args.param2)
