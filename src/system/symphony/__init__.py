# -*- coding: utf-8 -*-
"""
Symphony 系统包（按版本分子目录，避免边端启动时拉取 v1 的重依赖）：

- ``src.system.symphony.v1`` — benchmark / motivation（GOP 实验）
- ``src.system.symphony.v2`` — 边端入口（见 ``v2.edge``）

推荐显式导入，例如：
``from src.system.symphony.v1.benchmark import SymphonySystemBench``
``from src.system.symphony.v2.edge import SymphonySystemEdge, main``
"""
