from typing import TYPE_CHECKING

from typing import Hashable
from typing import List
from typing import Tuple
from typing import Optional
from typing import Dict

try:
    # Python 3.8 and newer
    from typing import Literal
    from typing import TypedDict
except ImportError:
    # Python 3.7 and older
    from typing_extensions import Literal
    from typing_extensions import TypedDict
