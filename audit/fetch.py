"""Download config, summary metrics and the final eval/front table for every public run in
openrlbenchmark/MORL-Baselines via the anonymous wandb GraphQL API.

Output: audit/data/runs.json (one record per run) and audit/data/fronts/<run>.json.
"""
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

API = "https://api.wandb.ai/graphql"
OUT = os.path.join(os.path.dirname(__file__), "data")


def curl(*args, tries=5):
    """Python's urllib gets its connection dropped here while curl works, so shell out."""
    for i in range(tries):
        r = subprocess.run(["curl", "-sSfL", "--max-time", "120", *args], capture_output=True)
        if r.returncode == 0:
            return r.stdout
        time.sleep(2 ** i)
    raise RuntimeError(r.stderr.decode())


def gql(query, variables=None):
    body = json.dumps(dict(query=query, variables=variables or {}))
    out = json.loads(curl(API, "-H", "Content-Type: application/json", "-d", body))
    if "errors" in out:
        raise RuntimeError(out["errors"])
    return out["data"]


RUNS = """query($after: String) { project(name: "MORL-Baselines", entityName: "openrlbenchmark") {
  runs(first: 100, after: $after) { pageInfo { hasNextPage endCursor }
    edges { node { name displayName state createdAt config summaryMetrics } } } } }"""

FILE = """query($run: String!, $path: String!) { project(name: "MORL-Baselines", entityName: "openrlbenchmark") {
  run(name: $run) { files(names: [$path]) { edges { node { directUrl } } } } } }"""


def fetch_runs():
    runs, after = [], None
    while True:
        page = gql(RUNS, dict(after=after))["project"]["runs"]
        for e in page["edges"]:
            n = e["node"]
            cfg = {k: v.get("value") if isinstance(v, dict) else v for k, v in json.loads(n["config"]).items()
                   if k != "_wandb"}
            runs.append(dict(name=n["name"], display=n["displayName"], state=n["state"],
                             created=n["createdAt"], config=cfg, summary=json.loads(n["summaryMetrics"])))
        if not page["pageInfo"]["hasNextPage"]:
            return runs
        after = page["pageInfo"]["endCursor"]


def fetch_front(run):
    dst = os.path.join(OUT, "fronts", run["name"] + ".json")
    front = run["summary"].get("eval/front")
    if os.path.exists(dst) or not isinstance(front, dict):
        return
    try:
        edges = gql(FILE, dict(run=run["name"], path=front["path"]))["project"]["run"]["files"]["edges"]
        data = curl(edges[0]["node"]["directUrl"])
        with open(dst, "wb") as f:
            f.write(data)
    except Exception as ex:  # keep going; missing fronts are reported later
        print(run["name"], "failed:", ex)


def main():
    os.makedirs(os.path.join(OUT, "fronts"), exist_ok=True)
    runs = fetch_runs()
    with open(os.path.join(OUT, "runs.json"), "w") as f:
        json.dump(runs, f)
    print(len(runs), "runs")
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(fetch_front, runs))
    print(len(os.listdir(os.path.join(OUT, "fronts"))), "fronts")


if __name__ == "__main__":
    main()
