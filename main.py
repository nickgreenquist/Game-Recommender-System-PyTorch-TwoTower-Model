"""
Game Recommender System — CLI entry point.

Usage:
    python main.py preprocess games         # Step 1: filter games → base_games, base_game_tags, base_vocab
    python main.py preprocess interactions  # Step 2: process user items → base_interactions_read/labels
    python main.py preprocess               # Run both steps in order
    python main.py explore                  # Explore user/game threshold distributions
    python main.py features                 # Stage 2: base parquets → data/features_*.parquet
    python main.py dataset                  # Stage 3: features → data/dataset_*_v1.pt
    python main.py train                    # Stage 4: train, save checkpoints (softmax)
    python main.py canary                   # Canary user recommendations (most recent checkpoint)
    python main.py canary <path>            # Canary user recommendations (specific checkpoint)
    python main.py probe                    # Embedding probes (most recent checkpoint)
    python main.py probe <path>             # Embedding probes (specific checkpoint)
    python main.py eval                     # Offline eval: Recall@K, NDCG@K, Hit Rate@K, MRR
    python main.py eval <path>              # Same, specific checkpoint
    python main.py export                   # Export serving artifacts for Streamlit
    python main.py export <path>            # Export using specific checkpoint
    python main.py                          # Run all stages in order
"""
import sys

DATA_DIR = 'data'
VERSION  = 'v1'


def cmd_preprocess(step=None):
    from src.preprocess import run
    run(data_dir=DATA_DIR, step=step)


def cmd_explore():
    from src.explore import run
    run(data_dir=DATA_DIR)


def cmd_features():
    from src.features import run
    run(data_dir=DATA_DIR, version=VERSION)


def cmd_dataset():
    from src.dataset import load_features, make_softmax_splits, save_softmax_splits
    print("Loading features ...")
    fs = load_features(DATA_DIR, VERSION)
    print("\nBuilding softmax datasets ...")
    train_data, val_data = make_softmax_splits(fs, DATA_DIR)
    save_softmax_splits(train_data, val_data, DATA_DIR, VERSION)


def cmd_train():
    from src.dataset import load_features, load_softmax_splits
    from src.train import get_config, build_model, train_softmax
    print("Loading features ...")
    fs = load_features(DATA_DIR, VERSION)
    print("\nLoading datasets ...")
    train_data, val_data = load_softmax_splits(DATA_DIR, VERSION)
    config = get_config()
    model  = build_model(config, fs)
    train_softmax(model, train_data, val_data, config, fs)


def cmd_canary(checkpoint_path=None):
    from src.evaluate import run_canary
    run_canary(data_dir=DATA_DIR, checkpoint_path=checkpoint_path, version=VERSION)


def cmd_probe(checkpoint_path=None):
    from src.evaluate import run_probes
    run_probes(data_dir=DATA_DIR, checkpoint_path=checkpoint_path, version=VERSION)


def cmd_eval(checkpoint_path=None):
    from src.dataset import load_features
    from src.evaluate import _resolve_checkpoint, _load_model_and_embeddings
    from src.offline_eval import run_offline_eval
    cp = _resolve_checkpoint(checkpoint_path, 'saved_models')
    if cp is None:
        return
    print("Loading features ...")
    fs = load_features(DATA_DIR, VERSION)
    model, _, _, _, _, _ = _load_model_and_embeddings(cp, fs)
    run_offline_eval(model, fs, checkpoint_path=cp)


def cmd_export(checkpoint_path=None):
    from src.export import run_export
    run_export(data_dir=DATA_DIR, checkpoint_path=checkpoint_path, version=VERSION)


COMMANDS = {
    'preprocess': cmd_preprocess,
    'explore':    cmd_explore,
    'features':   cmd_features,
    'dataset':    cmd_dataset,
    'train':      cmd_train,
    'canary':     cmd_canary,
    'probe':      cmd_probe,
    'eval':       cmd_eval,
    'export':     cmd_export,
}

if __name__ == '__main__':
    args = sys.argv[1:]

    if not args:
        print("Running all stages: preprocess → features → dataset → train → canary\n")
        cmd_preprocess()
        cmd_features()
        cmd_dataset()
        cmd_train()
        cmd_canary()
    elif args[0] == 'preprocess':
        step = args[1] if len(args) > 1 else None
        if step not in (None, 'games', 'interactions'):
            print("Usage: python main.py preprocess [games|interactions]")
            sys.exit(1)
        cmd_preprocess(step=step)
    elif args[0] in COMMANDS:
        if args[0] in ('canary', 'probe', 'export', 'eval') and len(args) > 1:
            COMMANDS[args[0]](checkpoint_path=args[1])
        else:
            COMMANDS[args[0]]()
    else:
        print(__doc__)
        sys.exit(1)
