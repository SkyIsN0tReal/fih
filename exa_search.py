from typing import Any, Dict, Optional

from exa_py import Exa
import os


IMAGE_LINKS_PER_RESULT = int(os.environ.get("EXA_IMAGE_LINKS_PER_RESULT", "5"))


exa = Exa(os.environ["EXA_API_KEY"])


def _build_extras(include_images: bool) -> Optional[Dict[str, Any]]:
    if not include_images:
        return None
    return {"imageLinks": IMAGE_LINKS_PER_RESULT}


def exa_search(query: str, *, include_images: bool = False):
    extras = _build_extras(include_images)
    kwargs: Dict[str, Any] = {
        "text": True,
        "type": "fast",
        "user_location": "US",
        "num_results": 2,
    }
    if extras is not None:
        kwargs["extras"] = extras
    result = exa.search_and_contents(
        query,
        **kwargs,
    )
    return result
