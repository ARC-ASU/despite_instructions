"""Follow-up security-game experiments reported in the camera-ready Appendices E and F.

Each study reuses the released incident protocol, prompts, monitor, and analysis
conventions, and changes one thing: more findings (``configs``), a report drawn
each round (``configs``), a sender without history (``stateless``), stronger
monitors (``monitors``), or stronger rewriters with a fidelity judge
(``rewriters``). ``analysis`` recomputes every reported number offline and
``replay`` checks the archived calls against exact request hashes.
"""
