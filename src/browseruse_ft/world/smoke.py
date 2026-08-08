"""Local execution-backed calibration for the browser reward contract."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from browseruse_ft.data import load_task_split

from .rollout import run_rollout, scripted_policy

PAGE = b"""<!doctype html><title>Sales report</title>
<h1>Admin dashboard</h1>
<button onclick="document.querySelector('#reports').hidden=false">Reports</button>
<section id="reports" hidden>
  <button onclick="document.querySelector('#sales').hidden=false">2022 best sellers</button>
</section>
<table id="sales" hidden>
  <tr><th>Rank</th><th>Product</th></tr>
  <tr><td>1</td><td>Quest Lumaflex&#8482; Band</td></tr>
</table>"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE)

    def log_message(self, _format: str, *_args) -> None:
        return


async def calibrate_rewards(output: Path) -> list[dict]:
    """Run success, failure, malformed, and reward-hacking probes."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        task = next(row for row in load_task_split("train") if row["task_id"] == 0)
        task = {
            **task,
            "start_url": f"http://127.0.0.1:{server.server_port}",
            "require_login": False,
        }
        answer = task["eval"]["reference_answers"]["exact_match"]
        candidates = {
            "known_good": [
                'do(action="Click", element="0")',
                'do(action="Click", element="1")',
                f'exit(message="{answer}")',
            ],
            "wrong_answer": ['exit(message="Not the product")'],
            "malformed_action": ["click 0", 'exit(message="Not the product")'],
            "answer_stuffing": [f'exit(message="The answer is {answer}")'],
            "invalid_then_correct": [
                'do(action="Click", element="999")',
                f'exit(message="{answer}")',
            ],
        }
        rows = []
        for label, actions in candidates.items():
            rollout = await run_rollout(task, scripted_policy(actions), max_steps=6)
            rows.append(
                {
                    "label": label,
                    "task_id": rollout["task_id"],
                    "terminal_score": rollout["terminal_score"],
                    "steps": len(rollout["steps"]),
                    "invalid_actions": sum(
                        not step["result"]["success"] for step in rollout["steps"]
                    ),
                    "reward_policy": rollout["reward_policy"],
                    "reward": rollout["reward"],
                    **rollout["rewards"],
                }
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return rows
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def run_calibration(output: Path) -> list[dict]:
    return asyncio.run(calibrate_rewards(output))
