#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Venus benchmark system：继承 VragSystemBench，场景聚类注入 + Venus 检索。"""

import logging
import os
import sys
from typing import Dict, Any, Optional

_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, _project_root)

from src.config import Config
from src.system.venus.motivation import VenusInjectRetrieveMixin
from src.system.vrag.benchmark import VragSystemBench


class VenusSystemBench(VenusInjectRetrieveMixin, VragSystemBench):
    """Benchmark：继承 VragSystemBench，覆盖 Venus 注入与检索。"""

    def __init__(self, config: Config = None):
        if config is None:
            config = Config()
        super().__init__(config)
        self._venus_enhance_doc = {}

    def _setup_logger(self):
        self.logger = logging.getLogger("VenusSystemBench")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(h)
        self.logger.propagate = False

    def _run_inject_phase(
        self,
        video_path: str,
        video_id: str,
        dataset_name: str,
        subset: Optional[str] = None,
        skip_inject: bool = False,
    ) -> Dict[str, Any]:
        return self._run_inject_phase_venus(
            video_path, video_id, dataset_name, subset, skip_if_exists=skip_inject
        )
