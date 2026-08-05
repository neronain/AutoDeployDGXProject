"""Fleet หลายเครื่อง — hub เครื่องเดียวคุม node อื่นผ่าน SSH

ต่างจาก stacked (หลายเครื่อง = โมเดลเดียว) ตรงที่แต่ละ node เป็นอิสระต่อกัน
node ล่มหนึ่งเครื่องไม่กระทบเครื่องอื่น
"""

from .cluster import (
    MIN_STACK_GBPS,
    check_cluster_ip,
    cluster_groups,
    cluster_note,
    fabric_links,
    stack_ready,
    suggest_cluster_ip,
)
from .registry import (
    Node,
    NodeError,
    add,
    find,
    load,
    nodes_file,
    remove,
    save,
    suggest_name,
    update,
    validate_cluster_ip,
)
from .ssh import check_login, ensure_key, install_key, key_path, probe, public_key_path, run

__all__ = [
    "add",
    "check_cluster_ip",
    "check_login",
    "cluster_groups",
    "cluster_note",
    "ensure_key",
    "fabric_links",
    "find",
    "install_key",
    "key_path",
    "load",
    "MIN_STACK_GBPS",
    "Node",
    "NodeError",
    "nodes_file",
    "probe",
    "public_key_path",
    "remove",
    "run",
    "save",
    "stack_ready",
    "suggest_cluster_ip",
    "suggest_name",
    "update",
    "validate_cluster_ip",
]
