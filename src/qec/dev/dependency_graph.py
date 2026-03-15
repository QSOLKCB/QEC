from __future__ import annotations

import ast
from collections import defaultdict, deque
from pathlib import Path
from typing import DefaultDict


class DependencyGraph:
    """Builds a deterministic import graph for modules in a package tree."""

    def __init__(self, package_root: Path, package_name: str = "src.qec") -> None:
        self.package_root = Path(package_root)
        self.package_name = package_name
        self.dependencies: dict[str, set[str]] = {}
        self.reverse_dependencies: dict[str, set[str]] = {}
        self._module_to_path: dict[str, Path] = {}

    def build(self) -> None:
        module_to_path = self._discover_modules()
        known_modules = set(module_to_path)
        dependencies: dict[str, set[str]] = {}
        reverse: DefaultDict[str, set[str]] = defaultdict(set)

        for module_name in sorted(module_to_path):
            module_path = module_to_path[module_name]
            imported = self._extract_dependencies(module_path, module_name, known_modules)
            dependencies[module_name] = imported
            for dep in sorted(imported):
                reverse[dep].add(module_name)

        for module_name in sorted(module_to_path):
            reverse.setdefault(module_name, set())

        self._module_to_path = module_to_path
        self.dependencies = dependencies
        self.reverse_dependencies = {k: set(v) for k, v in sorted(reverse.items())}

    def affected_modules(self, changed_modules: set[str]) -> list[str]:
        if not self.reverse_dependencies:
            self.build()

        queue = deque(sorted(changed_modules))
        visited = set(sorted(changed_modules))

        while queue:
            module = queue.popleft()
            for dependent in sorted(self.reverse_dependencies.get(module, set())):
                if dependent not in visited:
                    visited.add(dependent)
                    queue.append(dependent)

        return sorted(visited)

    def _discover_modules(self) -> dict[str, Path]:
        module_to_path: dict[str, Path] = {}
        for path in sorted(self.package_root.rglob("*.py")):
            module_name = self._module_name_from_path(path)
            if module_name is not None:
                module_to_path[module_name] = path
        return module_to_path

    def _module_name_from_path(self, path: Path) -> str | None:
        rel = path.relative_to(self.package_root)
        parts = list(rel.with_suffix("").parts)
        if not parts:
            return None
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            return self.package_name
        return ".".join([self.package_name, *parts])

    def _extract_dependencies(
        self,
        module_path: Path,
        module_name: str,
        known_modules: set[str],
    ) -> set[str]:
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))
        deps: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dep = self._best_known_module(alias.name, known_modules)
                    if dep is not None and dep != module_name:
                        deps.add(dep)
            elif isinstance(node, ast.ImportFrom):
                targets = self._from_import_targets(node, module_name, known_modules)
                for dep in sorted(targets):
                    if dep != module_name:
                        deps.add(dep)

        return deps

    def _from_import_targets(
        self,
        node: ast.ImportFrom,
        module_name: str,
        known_modules: set[str],
    ) -> set[str]:
        base = self._resolve_from_base(module_name, node.level, node.module)
        if base is None:
            return set()

        targets: set[str] = set()
        base_dep = self._best_known_module(base, known_modules)
        if base_dep is not None:
            targets.add(base_dep)

        for alias in node.names:
            if alias.name == "*":
                continue
            candidate = f"{base}.{alias.name}"
            cand_dep = self._best_known_module(candidate, known_modules)
            if cand_dep is not None:
                targets.add(cand_dep)

        return targets

    def _resolve_from_base(self, module_name: str, level: int, node_module: str | None) -> str | None:
        if level == 0:
            return node_module

        module_parts = module_name.split(".")
        package_parts = self.package_name.split(".")
        if len(module_parts) <= len(package_parts) or module_parts[: len(package_parts)] != package_parts:
            return None

        package_parts = module_parts[:-1]
        if level > 0:
            keep = len(package_parts) - (level - 1)
            if keep < 1:
                return None
            package_parts = package_parts[:keep]

        if node_module:
            return ".".join([*package_parts, node_module])

        return ".".join(package_parts)

    @staticmethod
    def _best_known_module(name: str, known_modules: set[str]) -> str | None:
        candidate = name
        while candidate:
            if candidate in known_modules:
                return candidate
            if "." not in candidate:
                break
            candidate = candidate.rsplit(".", 1)[0]
        return None
