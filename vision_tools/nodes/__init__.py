"""
Nodes package — pipeline node implementations.

Importing this package triggers auto-registration of all node types
with the NodeRegistry (via ``@NodeRegistry.register`` decorators).

Backend modules are also imported to register with BackendRegistry.
"""
# Import node modules to trigger @NodeRegistry.register decorators
from vision_tools.nodes import detection  # noqa: F401
from vision_tools.nodes import embedding  # noqa: F401
from vision_tools.nodes import captioning  # noqa: F401
from vision_tools.nodes import pose  # noqa: F401

# Import backend modules to trigger @BackendRegistry.register decorators
from vision_tools.backends.detection import yolo  # noqa: F401
from vision_tools.backends.embedding import siglip2  # noqa: F401
from vision_tools.backends.embedding import clip  # noqa: F401
from vision_tools.backends.captioning import smolvlm  # noqa: F401
from vision_tools.backends.captioning import llamacpp  # noqa: F401
from vision_tools.backends.pose import yolo_pose  # noqa: F401
