"""Interactive CLI console for OpenSI Cognitive Recurrence Runtime.

Allows a human user to:
- Assign tasks/goals interactively or via arguments.
- Observe internal cognitive recurrence steps in real-time.
- Respond to clarifying questions when the model emits ASK_USER.
- Approve or deny sensitive actions at policy approval gates.
- Gracefully pause/interrupt and resume sessions from disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cognitive_runtime.controller import CognitiveController
from cognitive_runtime.state import StateStore, new_state
from harness_adapter import HarnessAdapter
from model_adapter import get_backend


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class InteractiveConsole:
    def __init__(self, config: dict[str, Any], auto_approve: bool = False):
        self.config = config
        self.auto_approve = auto_approve
        self.harness = HarnessAdapter(config)
        self.backend = get_backend(config.get("backend", "mock"), config)
        self.store = StateStore(ROOT / "data" / "sessions")
        self.controller = CognitiveController(
            backend=self.backend,
            gateway=self.harness.gateway,
            config=config,
            trace_dir=str(ROOT / "data" / "traces"),
            approval_handler=self.handle_approval,
            ask_user_handler=self.handle_ask_user,
            step_callback=self.handle_step,
            store=self.store,
        )

    def handle_step(self, step_info: dict[str, Any]) -> None:
        step = step_info["step"]
        action = step_info["action"]
        dec = step_info.get("decision", {})
        rationale = dec.get("rationale", "")

        prefix = f"\033[1;34m[Step {step}]\033[0m"
        action_colored = f"\033[1;33m{action}\033[0m"

        print(f"\n{prefix} Action: {action_colored}")
        if rationale:
            print(f"       \033[2mRationale: {rationale}\033[0m")

        if action == "CALL_TOOL":
            tool_name = dec.get("tool", "")
            args = dec.get("arguments", {})
            print(f"       \033[36mCalling Tool:\033[0m {tool_name} {args}")

    def handle_approval(self, decision: dict[str, Any]) -> bool:
        tool_name = decision.get("tool", "")
        args = decision.get("arguments", {})
        print(f"\n\033[1;31m[SECURITY APPROVAL GATE]\033[0m")
        print(f"Tool: \033[1m{tool_name}\033[0m")
        print(f"Arguments: {args}")

        if self.auto_approve:
            print("Auto-approved via --auto-approve.")
            return True

        try:
            choice = input("Approve tool execution? [y/N]: ").strip().lower()
            return choice in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            print("\nDenied.")
            return False

    def handle_ask_user(self, question: str) -> str:
        print(f"\n\033[1;35m[MODEL ASKS USER]\033[0m")
        print(f"\033[1m{question}\033[0m")
        try:
            answer = input("Your answer: ").strip()
            return answer
        except (KeyboardInterrupt, EOFError):
            print("\nNo reply given.")
            return ""

    def run(self, goal: str | None = None, resume_id: str | None = None) -> int:
        print("\033[1;32m====================================================\033[0m")
        print("\033[1;32m    OpenSI Cognitive Recurrence Interactive Console \033[0m")
        print(f"\033[2m    Backend: {self.config.get('backend')} | Sandbox: {self.config.get('policy', {}).get('sandbox_roots')}\033[0m")
        print("\033[1;32m====================================================\033[0m")

        state = None
        if resume_id:
            if not self.store.exists(resume_id):
                print(f"Error: Session {resume_id} not found in {self.store.directory}", file=sys.stderr)
                return 1
            print(f"Resuming session: {resume_id}")
            state = self.store.load(resume_id)
        elif not goal:
            try:
                goal = input("\nEnter objective / goal: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                return 0

        if not state:
            state = new_state(goal=goal)

        print(f"\nActive Session ID: \033[1m{state.session_id}\033[0m")
        print(f"Goal: \033[1m{state.goal}\033[0m\n")

        try:
            outcome = self.controller.run(state)
        except KeyboardInterrupt:
            print("\n\n\033[1;33mExecution paused by user (Ctrl+C).\033[0m")
            self.store.save(state)
            print(f"Session saved. You can resume using:")
            print(f"  python scripts/interactive_cli.py --resume {state.session_id}")
            return 0

        print("\n\033[1;32m====================================================\033[0m")
        print(f"\033[1;32m[COMPLETED]\033[0m Stop Reason: \033[1m{outcome.stop_reason}\033[0m")
        print(f"Steps: {outcome.steps} | Tool calls: {outcome.tool_calls} | Time: {outcome.seconds:.2f}s")
        if outcome.final_text:
            print(f"\n\033[1mFinal Answer:\033[0m\n{outcome.final_text}")
        print("\033[1;32m====================================================\033[0m")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenSI Interactive CLI Console")
    parser.add_argument("--config", "-c", default=str(ROOT / "configs" / "prototype.yaml"),
                        help="Path to YAML configuration")
    parser.add_argument("--backend", "-b", help="Override inference backend")
    parser.add_argument("--goal", "-g", help="Objective/goal to execute")
    parser.add_argument("--resume", "-r", help="Session ID to resume")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve sensitive tool calls")

    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    if args.backend:
        cfg["backend"] = args.backend

    console = InteractiveConsole(cfg, auto_approve=args.auto_approve)
    return console.run(goal=args.goal, resume_id=args.resume)


if __name__ == "__main__":
    sys.exit(main())
