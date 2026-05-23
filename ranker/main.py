"""
Ranker pipeline entry point.

Usage:
    python ranker/main.py precompute                # Stage 0: generate candidates (default PROD CG α=0.4)
    python ranker/main.py precompute <cg.pth>       # Stage 0: override CG checkpoint
    python ranker/main.py train                     # Stage 1+: train Wide & Deep ranker
    python ranker/main.py evaluate                  # Eval-only: auto-find most recent ranker checkpoint
    python ranker/main.py evaluate <ranker.pth>     # Eval-only: explicit checkpoint
    python ranker/main.py canary                    # Top-N for ALL canaries → ranker/canary_results/<ckpt>.txt
    python ranker/main.py canary <ranker.pth>       # Same, explicit checkpoint
    python ranker/main.py export                    # Export serving artifacts (α=0 CG + latest ranker)
    python ranker/main.py export <ranker.pth>       # Same, explicit ranker checkpoint
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranker.canary import dump_canary
from ranker.export import run_ranker_export
from ranker.precompute import precompute
from ranker.train import evaluate_only, train


USAGE = __doc__.strip()


def main():
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'precompute':
        ckpt = sys.argv[2] if len(sys.argv) > 2 else None
        precompute(checkpoint_path=ckpt)
    elif cmd == 'train':
        train()
    elif cmd == 'evaluate':
        ckpt = sys.argv[2] if len(sys.argv) > 2 else None
        evaluate_only(ckpt)
    elif cmd == 'canary':
        ckpt = sys.argv[2] if len(sys.argv) > 2 else None
        dump_canary(ranker_checkpoint=ckpt)
    elif cmd == 'export':
        ckpt = sys.argv[2] if len(sys.argv) > 2 else None
        run_ranker_export(ranker_checkpoint=ckpt)
    else:
        print(f"Unknown command: {cmd}\n")
        print(USAGE)
        sys.exit(1)


if __name__ == '__main__':
    main()
