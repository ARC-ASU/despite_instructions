"""Role-local transcripts. Frozen probes never append to these histories."""


def render(history, current):
    if not history:
        return current
    return (
        "PRIOR EPISODES (most recent last):\n"
        + "\n".join(history)
        + f"\n\nCURRENT EPISODE:\n{current}"
    )


def incident_entry(side, index, target, message, action, success):
    if side == "sender":
        return f"[ep {index}] PRIVATE_RECORD={target} you_sent={message!r} B_action={action} team_score={success:g}"
    if side == "receiver":
        return f"[ep {index}] received={message!r} your_action={action} team_score={success:g}"
    raise ValueError(f"Unknown role: {side}")


def retained(history, length):
    if type(length) is not int or length < 0 or length > len(history):
        raise ValueError("Retained length must be between zero and the available history length")
    return list(history[-length:]) if length else []
