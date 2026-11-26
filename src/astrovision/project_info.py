"""Print a short status message for the AstroVision project."""

PROJECT_NAME: str = "AstroVision"
CURRENT_STAGE: str = "Python foundation"


def make_status_message(project_name: str, stage: str) -> str:
    """Build and return a readable project status message."""
    message = f"{project_name}: {stage} is ready."
    return message


def main() -> None:
    """Run this module's small demonstration."""
    status_message = make_status_message(PROJECT_NAME, CURRENT_STAGE)
    print(status_message)


if __name__ == "__main__":
    main()
