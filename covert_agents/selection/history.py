"""Historical sentence-selection transcript format."""


def render(history_blocks, core):
    if not history_blocks:
        return core
    return (
        "PRIOR EPISODES (most recent last):\n"
        + "\n".join(history_blocks)
        + f"\n\nCURRENT EPISODE:\n{core}"
    )
