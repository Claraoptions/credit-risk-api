"""Train the model and save it: ``python -m app.train``."""

from . import model as m


def main() -> None:
    trained = m.train()
    path = m.save(trained)
    print(f"Saved model v{trained.version} to {path}")
    for k, v in trained.metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
