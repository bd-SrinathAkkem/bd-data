#!/usr/bin/env python3
"""
Smart Repository Build Command Analyzer

Copyright (c) 2025 Black Duck Software Inc.
Author: Srinath Akkem <reddyakkem@example.com>
License: MIT

Analyzes repositories to generate accurate build commands for various project types.
Follows PEP 8 conventions with clear naming and robust error handling.
Optimized for mixed codebases and subprojects in 2025.
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import shutil
import tempfile

import requests
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class BuildCommand:
    """Represents a build command with its metadata."""
    command: str
    confidence: int
    source: str
    description: str
    category: str
    priority: int = 0
    validation_result: str = "unknown"
    requires: List[str] = field(default_factory=list)

@dataclass
class ProjectInfo:
    """Holds comprehensive project analysis information."""
    project_type: str
    primary_language: str
    languages: Dict[str, float]
    build_system: str
    package_manager: str
    frameworks: List[str]
    build_commands: List[BuildCommand]
    analysis_confidence: float
    project_structure: Dict[str, Any]
    entry_points: List[str]
    errors: List[str]
    subprojects: List[Dict[str, Any]] = field(default_factory=list)

class SafeFileParser:
    """Provides robust file parsing with error handling."""

    @staticmethod
    def read_file_safe(file_path: Path, max_size: int = 200_000) -> Optional[str]:
        """Safely reads file content with encoding detection."""
        if not file_path.exists() or not file_path.is_file():
            return None
        try:
            if file_path.stat().st_size > max_size:
                logger.debug(f"File too large: {file_path}")
                return None
            for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
                try:
                    content = file_path.read_text(encoding=encoding)
                    if "\x00" in content:
                        return None
                    return content
                except (UnicodeDecodeError, UnicodeError):
                    continue
        except Exception as e:
            logger.debug(f"Error reading {file_path}: {e}")
        return None

    @staticmethod
    def parse_json_robust(content: str) -> Optional[Dict]:
        """Parses JSON content, handling common issues."""
        if not content or not content.strip():
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                fixed = re.sub(r",(\s*[}\]])", r"\1", content)
                fixed = re.sub(r"//.*$", "", fixed, flags=re.MULTILINE)
                fixed = re.sub(r"/\*.*?\*/", "", fixed, flags=re.DOTALL)
                return json.loads(fixed)
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parsing failed: {e}")
                return None

    @staticmethod
    def parse_xml_safe(content: str) -> Optional[ET.Element]:
        """Safely parses XML content."""
        if not content or not content.strip():
            return None
        try:
            return ET.fromstring(content)
        except ET.ParseError:
            try:
                cleaned = re.sub(r"<\?xml[^>]*\?>", "", content)
                return ET.fromstring(cleaned)
            except ET.ParseError as e:
                logger.debug(f"XML parsing failed: {e}")
                return None

    @staticmethod
    def parse_yaml_safe(content: str) -> Optional[Dict]:
        """Safely parses YAML content."""
        if not content or not content.strip():
            return None
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError as e:
            logger.debug(f"YAML parsing failed: {e}")
            return None

class ProjectTypeDetector:
    """Detects project types and subprojects with high accuracy."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.parser = SafeFileParser()

    def detect_projects(self) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Detects multiple project types in the repository."""
        detectors = [
            self._detect_nodejs,
            self._detect_spring_boot,
            self._detect_java_maven,
            self._detect_java_gradle,
            self._detect_python,
            self._detect_go,
            self._detect_generic_java,
        ]
        results = [detector() for detector in detectors if detector()[1] > 50]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def find_subprojects(self) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Detects subprojects in top-level directories."""
        subprojects = []
        ignore_dirs = {".git", "node_modules", "__pycache__", "venv", "target", "build", "dist"}

        for subdir in self.repo_path.iterdir():
            if subdir.is_dir() and subdir.name not in ignore_dirs and not subdir.name.startswith("."):
                sub_detector = ProjectTypeDetector(subdir)
                sub_detections = sub_detector.detect_projects()
                if sub_detections and sub_detections[0][1] > 70:
                    subprojects.append((subdir.name, *sub_detections[0]))

        return subprojects

    def _detect_nodejs(self) -> Tuple[str, float, Dict[str, Any]]:
        """Detects Node.js projects."""
        confidence = 0.0
        analysis: Dict[str, Any] = {"project_type": "nodejs"}

        package_json = self.repo_path / "package.json"
        if package_json.exists():
            content = self.parser.read_file_safe(package_json)
            if content:
                package_data = self.parser.parse_json_robust(content)
                if package_data:
                    confidence = 95.0
                    analysis.update({
                        "package_json": package_data,
                        "scripts": package_data.get("scripts", {}),
                        "dependencies": package_data.get("dependencies", {}),
                        "dev_dependencies": package_data.get("devDependencies", {}),
                        "main": package_data.get("main", "index.js"),
                        "engines": package_data.get("engines", {}),
                        "type": package_data.get("type", "commonjs"),
                    })
                    all_deps = {**package_data.get("dependencies", {}), **package_data.get("devDependencies", {})}
                    frameworks = []
                    for dep, framework in [
                        ("react", "react"), ("vue", "vue"), ("@angular/core", "angular"),
                        ("next", "nextjs"), ("express", "express"), ("@nestjs/core", "nestjs")
                    ]:
                        if dep in all_deps or f"@types/{dep}" in all_deps:
                            frameworks.append(framework)
                    analysis["frameworks"] = frameworks
                else:
                    confidence = 60.0
                    analysis["parse_error"] = True

        lock_files = {
            "yarn.lock": "yarn", "pnpm-lock.yaml": "pnpm",
            "package-lock.json": "npm", "bun.lockb": "bun"
        }
        for lock_file, manager in lock_files.items():
            if (self.repo_path / lock_file).exists():
                confidence += 5
                analysis["package_manager"] = manager
                break
        else:
            analysis["package_manager"] = "npm"

        if (self.repo_path / "tsconfig.json").exists():
            confidence += 10
            analysis["typescript"] = True

        for file in ["server.js", "app.js", "index.js", ".nvmrc", "nodemon.json"]:
            if (self.repo_path / file).exists():
                confidence += 3

        return "nodejs", min(confidence, 98.0), analysis

    def _detect_spring_boot(self) -> Tuple[str, float, Dict[str, Any]]:
        """Detects Spring Boot projects."""
        confidence = 0.0
        analysis: Dict[str, Any] = {"project_type": "spring_boot"}

        pom_xml = self.repo_path / "pom.xml"
        if pom_xml.exists():
            content = self.parser.read_file_safe(pom_xml)
            if content and ("spring-boot-starter-parent" in content or "spring-boot-starter" in content):
                confidence = 95.0
                analysis["build_system"] = "maven"
                xml_root = self.parser.parse_xml_safe(content)
                if xml_root is not None:
                    parent = xml_root.find(".//{http://maven.apache.org/POM/4.0.0}parent")
                    if parent is not None:
                        version_elem = parent.find(".//{http://maven.apache.org/POM/4.0.0}version")
                        if version_elem is not None:
                            analysis["spring_boot_version"] = version_elem.text
                    artifact_id = xml_root.find(".//{http://maven.apache.org/POM/4.0.0}artifactId")
                    if artifact_id is not None:
                        analysis["artifact_id"] = artifact_id.text

        build_gradle = self.repo_path / "build.gradle"
        if build_gradle.exists() and confidence < 50:
            content = self.parser.read_file_safe(build_gradle)
            if content and ("org.springframework.boot" in content or "spring-boot" in content):
                confidence = 90.0
                analysis["build_system"] = "gradle"

        src_main_java = self.repo_path / "src" / "main" / "java"
        if src_main_java.exists() and confidence > 50:
            for java_file in src_main_java.rglob("*.java"):
                content = self.parser.read_file_safe(java_file)
                if content and "@SpringBootApplication" in content:
                    confidence += 10
                    analysis["main_class"] = java_file.stem
                    break

        for config in ["application.properties", "application.yml"]:
            if (self.repo_path / "src" / "main" / "resources" / config).exists():
                confidence += 5
                analysis["has_app_config"] = True

        return "spring_boot", min(confidence, 98.0), analysis

    def _detect_java_maven(self) -> Tuple[str, float, Dict[str, Any]]:
        """Detects Maven-based Java projects."""
        confidence = 0.0
        analysis: Dict[str, Any] = {"project_type": "java_maven"}

        pom_xml = self.repo_path / "pom.xml"
        if pom_xml.exists():
            content = self.parser.read_file_safe(pom_xml)
            if content:
                xml_root = self.parser.parse_xml_safe(content)
                if xml_root is not None:
                    confidence = 85.0
                    analysis["build_system"] = "maven"
                    try:
                        for elem, key in [
                            (".//{http://maven.apache.org/POM/4.0.0}groupId", "group_id"),
                            (".//{http://maven.apache.org/POM/4.0.0}artifactId", "artifact_id"),
                            (".//{http://maven.apache.org/POM/4.0.0}version", "version"),
                            (".//{http://maven.apache.org/POM/4.0.0}properties/{http://maven.apache.org/POM/4.0.0}maven.compiler.source", "java_version")
                        ]:
                            found = xml_root.find(elem)
                            if found is not None:
                                analysis[key] = found.text
                    except Exception as e:
                        logger.debug(f"Error parsing Maven POM: {e}")
                else:
                    confidence = 70.0

        if (self.repo_path / "src" / "main" / "java").exists():
            confidence += 10
            analysis["standard_structure"] = True

        return "java_maven", min(confidence, 95.0), analysis

    def _detect_java_gradle(self) -> Tuple[str, float, Dict[str, Any]]:
        """Detects Gradle-based Java projects."""
        confidence = 0.0
        analysis: Dict[str, Any] = {"project_type": "java_gradle"}

        for gradle_file in ["build.gradle", "build.gradle.kts"]:
            if (self.repo_path / gradle_file).exists():
                content = self.parser.read_file_safe(self.repo_path / gradle_file)
                if content:
                    confidence = 85.0
                    analysis["build_system"] = "gradle"
                    analysis["gradle_file"] = gradle_file
                    if "java" in content or "org.gradle.api.plugins.JavaPlugin" in content:
                        confidence += 5
                    if (self.repo_path / "gradlew").exists():
                        confidence += 5
                        analysis["has_wrapper"] = True
                    break

        return "java_gradle", min(confidence, 95.0), analysis

    def _detect_python(self) -> Tuple[str, float, Dict[str, Any]]:
        """Detects Python projects."""
        confidence = 0.0
        analysis: Dict[str, Any] = {"project_type": "python"}

        python_configs = {
            "pyproject.toml": 95, "setup.py": 90, "setup.cfg": 85,
            "requirements.txt": 70, "Pipfile": 85, "poetry.lock": 90,
            "conda.yml": 80, "environment.yml": 80
        }

        for config_file, conf_boost in python_configs.items():
            if (self.repo_path / config_file).exists():
                content = self.parser.read_file_safe(self.repo_path / config_file)
                if content:
                    confidence = max(confidence, conf_boost)
                    analysis[config_file.replace(".", "_")] = True
                    if config_file == "pyproject.toml":
                        toml_data = self.parser.parse_yaml_safe(content)
                        if toml_data and "build-system" in toml_data:
                            analysis["build_system"] = "pip"
                            if "poetry" in str(toml_data.get("build-system", {})):
                                analysis["package_manager"] = "poetry"
                    elif config_file == "setup.py":
                        if "setuptools" in content or "distutils" in content:
                            analysis["build_system"] = "setuptools"

        for entry in ["main.py", "app.py", "run.py", "server.py", "manage.py", "__main__.py"]:
            if (self.repo_path / entry).exists():
                confidence += 5
                analysis["entry_points"] = analysis.get("entry_points", []) + [entry]

        if (self.repo_path / "__init__.py").exists():
            confidence += 10
            analysis["is_package"] = True

        frameworks = []
        if (self.repo_path / "manage.py").exists():
            frameworks.append("django")
        if any((self.repo_path / f).exists() for f in ["app.py", "application.py"]):
            req_file = self.repo_path / "requirements.txt"
            if req_file.exists():
                req_content = self.parser.read_file_safe(req_file)
                if req_content:
                    for framework in ["flask", "fastapi", "django"]:
                        if framework in req_content.lower():
                            frameworks.append(framework)
        analysis["frameworks"] = frameworks

        return "python", min(confidence, 95.0), analysis

    def _detect_go(self) -> Tuple[str, float, Dict[str, Any]]:
        """Detects Go projects."""
        confidence = 0.0
        analysis: Dict[str, Any] = {"project_type": "go"}

        go_mod = self.repo_path / "go.mod"
        if go_mod.exists():
            content = self.parser.read_file_safe(go_mod)
            if content:
                confidence = 95.0
                analysis["has_go_mod"] = True
                module_match = re.search(r"module\s+([^\s\n]+)", content)
                if module_match:
                    analysis["module_name"] = module_match.group(1)
                go_version_match = re.search(r"go\s+([\d.]+)", content)
                if go_version_match:
                    analysis["go_version"] = go_version_match.group(1)

        main_go = self.repo_path / "main.go"
        if main_go.exists():
            confidence = max(confidence, 80.0)
            analysis["has_main_go"] = True
            content = self.parser.read_file_safe(main_go)
            if content and "func main()" in content:
                confidence += 5

        if (self.repo_path / "go.sum").exists():
            confidence += 5
            analysis["has_go_sum"] = True

        for go_dir in ["cmd", "pkg", "internal", "api", "web"]:
            if (self.repo_path / go_dir).is_dir():
                confidence += 3

        return "go", min(confidence, 98.0), analysis

    def _detect_generic_java(self) -> Tuple[str, float, Dict[str, Any]]:
        """Detects generic Java projects as a fallback."""
        confidence = 0.0
        analysis: Dict[str, Any] = {"project_type": "java"}

        java_files = list(self.repo_path.rglob("*.java"))
        if java_files:
            confidence = 60.0
            analysis["java_files_count"] = len(java_files)

        if (self.repo_path / "src").exists():
            confidence += 10

        return "java", confidence, analysis

class SmartCommandGenerator:
    """Generates and validates build commands based on project analysis."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.parser = SafeFileParser()

    def generate_commands(self, project_type: str, analysis_data: Dict[str, Any]) -> List[BuildCommand]:
        """Generates build commands for the detected project type."""
        generators = {
            "nodejs": self._generate_nodejs_commands,
            "spring_boot": self._generate_spring_boot_commands,
            "java_maven": self._generate_maven_commands,
            "java_gradle": self._generate_gradle_commands,
            "python": self._generate_python_commands,
            "go": self._generate_go_commands,
            "java": self._generate_generic_java_commands,
        }
        generator = generators.get(project_type, lambda _: [])
        commands = generator(analysis_data)

        for command in commands:
            self._validate_tool_availability(command)

        commands.sort(key=lambda x: (x.priority, x.confidence), reverse=True)
        return commands

    def _generate_nodejs_commands(self, analysis: Dict[str, Any]) -> List[BuildCommand]:
        """Generates commands for Node.js projects."""
        commands = []
        pkg_manager = analysis.get("package_manager", "npm")

        install_cmd = {
            "npm": "npm install", "yarn": "yarn install",
            "pnpm": "pnpm install", "bun": "bun install"
        }
        commands.append(BuildCommand(
            command=install_cmd.get(pkg_manager, "npm install"),
            confidence=95,
            source="package_manager",
            description=f"Install dependencies using {pkg_manager}",
            category="install",
            priority=100,
            requires=["package.json"],
        ))

        scripts = analysis.get("scripts", {})
        script_priorities = {
            "build": (90, "build"), "start": (85, "run"), "dev": (80, "dev"),
            "develop": (80, "dev"), "serve": (75, "dev"), "test": (70, "test"),
            "lint": (65, "lint"), "format": (60, "lint")
        }

        for script_name, script_cmd in scripts.items():
            priority, category = script_priorities.get(script_name, (50, "other"))
            cmd = f"{pkg_manager} run {script_name}" if pkg_manager != "yarn" or script_name not in ["start", "test"] else f"yarn {script_name}"
            commands.append(BuildCommand(
                command=cmd,
                confidence=85 if script_name in script_priorities else 70,
                source="package.json",
                description=f"Run {script_name}: {script_cmd[:50]}..." if len(script_cmd) > 50 else f"Run {script_name}: {script_cmd}",
                category=category,
                priority=priority,
                requires=["package.json"],
            ))

        if analysis.get("typescript"):
            commands.append(BuildCommand(
                command="npx tsc",
                confidence=75,
                source="typescript",
                description="Compile TypeScript",
                category="build",
                priority=75,
                requires=["tsconfig.json"],
            ))

        frameworks = analysis.get("frameworks", [])
        if "react" in frameworks and "build" not in scripts:
            commands.append(BuildCommand(
                command=f"{pkg_manager} run build",
                confidence=70,
                source="react",
                description="Build React app",
                category="build",
                priority=70,
            ))

        if "nextjs" in frameworks:
            for cmd, conf, desc, cat, prio in [
                ("next build", 80, "Build Next.js app", "build", 80),
                ("next start", 75, "Start Next.js production server", "run", 75),
                ("next dev", 85, "Start Next.js development server", "dev", 85),
            ]:
                commands.append(BuildCommand(
                    command=cmd, confidence=conf, source="nextjs", description=desc, category=cat, priority=prio
                ))

        return commands

    def _generate_spring_boot_commands(self, analysis: Dict[str, Any]) -> List[BuildCommand]:
        """Generates commands for Spring Boot projects."""
        commands = []
        build_system = analysis.get("build_system", "maven")

        if build_system == "maven":
            maven_commands = [
                ("mvn clean install", 95, "Clean and build Spring Boot app", "build", 95),
                ("mvn spring-boot:run", 90, "Run Spring Boot application", "run", 90),
                ("mvn test", 80, "Run tests", "test", 80),
                ("mvn compile", 75, "Compile source code", "build", 75),
                ("mvn package", 85, "Package application", "build", 85),
                ("mvn clean", 70, "Clean build artifacts", "clean", 70),
            ]
        else:
            wrapper = "./gradlew" if analysis.get("has_wrapper") else "gradle"
            maven_commands = [
                (f"{wrapper} build", 95, "Build Spring Boot app", "build", 95),
                (f"{wrapper} bootRun", 90, "Run Spring Boot application", "run", 90),
                (f"{wrapper} test", 80, "Run tests", "test", 80),
                (f"{wrapper} assemble", 75, "Assemble application", "build", 75),
                (f"{wrapper} clean", 70, "Clean build artifacts", "clean", 70),
            ]

        for cmd, conf, desc, cat, prio in maven_commands:
            commands.append(BuildCommand(
                command=cmd,
                confidence=conf,
                source=build_system,
                description=desc,
                category=cat,
                priority=prio,
                requires=["pom.xml"] if build_system == "maven" else ["build.gradle"],
            ))

        return commands

    def _generate_maven_commands(self, analysis: Dict[str, Any]) -> List[BuildCommand]:
        """Generates commands for Maven-based Java projects."""
        commands = []
        maven_commands = [
            ("mvn clean install", 95, "Clean and install project", "build", 95),
            ("mvn compile", 85, "Compile source code", "build", 85),
            ("mvn test", 80, "Run unit tests", "test", 80),
            ("mvn package", 90, "Package compiled code", "build", 90),
            ("mvn clean", 70, "Clean target directory", "clean", 70),
            ("mvn exec:java", 70, "Execute main class", "run", 70),
            ("mvn dependency:resolve", 75, "Download dependencies", "install", 75),
        ]

        for cmd, conf, desc, cat, prio in maven_commands:
            commands.append(BuildCommand(
                command=cmd,
                confidence=conf,
                source="maven",
                description=desc,
                category=cat,
                priority=prio,
                requires=["pom.xml"],
            ))

        return commands

    def _generate_gradle_commands(self, analysis: Dict[str, Any]) -> List[BuildCommand]:
        """Generates commands for Gradle-based Java projects."""
        commands = []
        wrapper = "./gradlew" if analysis.get("has_wrapper") else "gradle"

        gradle_commands = [
            (f"{wrapper} build", 95, "Build the project", "build", 95),
            (f"{wrapper} assemble", 85, "Assemble project outputs", "build", 85),
            (f"{wrapper} test", 80, "Run unit tests", "test", 80),
            (f"{wrapper} clean", 70, "Clean build directory", "clean", 70),
            (f"{wrapper} run", 75, "Run the application", "run", 75),
            (f"{wrapper} dependencies", 65, "Display dependencies", "install", 65),
        ]

        for cmd, conf, desc, cat, prio in gradle_commands:
            commands.append(BuildCommand(
                command=cmd,
                confidence=conf,
                source="gradle",
                description=desc,
                category=cat,
                priority=prio,
                requires=[analysis.get("gradle_file", "build.gradle")],
            ))

        return commands

    def _generate_python_commands(self, analysis: Dict[str, Any]) -> List[BuildCommand]:
        """Generates commands for Python projects."""
        commands = []

        if analysis.get("poetry_lock"):
            commands.extend([
                BuildCommand("poetry install", 95, "Install dependencies with Poetry", "install", 100, source="poetry"),
                BuildCommand("poetry run pytest", 80, "Run tests with Poetry", "test", 75, source="poetry"),
                BuildCommand("poetry build", 85, "Build package with Poetry", "build", 80, source="poetry"),
                BuildCommand("poetry shell", 70, "Activate Poetry virtual environment", "dev", 70, source="poetry"),
            ])

        elif analysis.get("pipfile"):
            commands.extend([
                BuildCommand("pipenv install", 90, "Install dependencies with Pipenv", "install", 95, source="pipenv"),
                BuildCommand("pipenv install --dev", 85, "Install dev dependencies", "install", 85, source="pipenv"),
                BuildCommand("pipenv run pytest", 80, "Run tests with Pipenv", "test", 75, source="pipenv"),
                BuildCommand("pipenv shell", 70, "Activate Pipenv shell", "dev", 70, source="pipenv"),
            ])

        elif analysis.get("requirements_txt"):
            commands.extend([
                BuildCommand("pip install -r requirements.txt", 90, "Install from requirements.txt", "install", 95, source="pip"),
                BuildCommand("python -m venv venv", 85, "Create virtual environment", "install", 90, source="pip"),
                BuildCommand("source venv/bin/activate && pip install -r requirements.txt", 85, "Setup venv and install deps", "install", 85, source="pip"),
            ])

        elif analysis.get("setup_py"):
            commands.extend([
                BuildCommand("pip install -e .", 85, "Install package in development mode", "install", 85, source="setup.py"),
                BuildCommand("pip install .", 80, "Install package", "install", 80, source="setup.py"),
                BuildCommand("python setup.py build", 75, "Build package", "build", 75, source="setup.py"),
            ])

        elif analysis.get("pyproject_toml"):
            commands.extend([
                BuildCommand("pip install -e .", 85, "Install in development mode", "install", 85, source="pyproject.toml"),
                BuildCommand("pip install .", 80, "Install package", "install", 80, source="pyproject.toml"),
            ])

        entry_points = analysis.get("entry_points", [])
        for entry in entry_points:
            priority = {"main.py": 90, "app.py": 85, "run.py": 80, "server.py": 85, "manage.py": 85}.get(entry, 70)
            commands.append(BuildCommand(
                command=f"python {entry}",
                confidence=85,
                source="entry_point",
                description=f"Run {entry}",
                category="run",
                priority=priority,
                requires=[entry],
            ))

        frameworks = analysis.get("frameworks", [])
        if "django" in frameworks:
            commands.extend([
                BuildCommand("python manage.py runserver", 90, "Start Django development server", "run", 85, source="django"),
                BuildCommand("python manage.py migrate", 80, "Run Django migrations", "install", 75, source="django"),
                BuildCommand("python manage.py collectstatic", 75, "Collect static files", "build", 70, source="django"),
            ])

        if "flask" in frameworks:
            commands.extend([
                BuildCommand("flask run", 85, "Start Flask development server", "run", 80, source="flask"),
                BuildCommand("python -m flask run", 85, "Start Flask server (alternative)", "run", 80, source="flask"),
            ])

        if "fastapi" in frameworks:
            commands.extend([
                BuildCommand("uvicorn main:app --reload", 85, "Start FastAPI with auto-reload", "run", 85, source="fastapi"),
                BuildCommand("python -m uvicorn main:app", 80, "Start FastAPI server", "run", 80, source="fastapi"),
            ])

        test_commands = [
            ("pytest", 85, "Run tests with pytest", "test", 70, "python_testing"),
            ("python -m unittest discover", 75, "Run tests with unittest", "test", 65, "python_testing"),
            ("python -m doctest", 60, "Run doctests", "test", 60, "python_testing"),
        ]

        for cmd, conf, desc, cat, prio, src in test_commands:
            commands.append(BuildCommand(
                command=cmd, confidence=conf, source=src, description=desc, category=cat, priority=prio
            ))

        return commands

    def _generate_go_commands(self, analysis: Dict[str, Any]) -> List[BuildCommand]:
        """Generates commands for Go projects."""
        commands = []

        if analysis.get("has_go_mod"):
            go_commands = [
                ("go mod download", 95, "Download dependencies", "install", 100),
                ("go mod tidy", 90, "Clean up dependencies", "install", 95),
                ("go build", 90, "Build the application", "build", 90),
                ("go build -o app", 85, "Build with specific output name", "build", 85),
                ("go install", 80, "Install the application", "install", 80),
                ("go test", 85, "Run tests", "test", 85),
                ("go test ./...", 80, "Run all tests recursively", "test", 80),
                ("go run .", 85, "Run the current package", "run", 85),
                ("go fmt ./...", 70, "Format all Go code", "lint", 70),
                ("go vet ./...", 70, "Vet Go code for issues", "lint", 70),
                ("go mod verify", 65, "Verify dependencies", "install", 65),
            ]
            if analysis.get("has_main_go"):
                go_commands.insert(5, ("go run main.go", 85, "Run main.go directly", "run", 85))

            for cmd, conf, desc, cat, prio in go_commands:
                commands.append(BuildCommand(
                    command=cmd,
                    confidence=conf,
                    source="go",
                    description=desc,
                    category=cat,
                    priority=prio,
                    requires=["go.mod"] if "go mod" in cmd else [],
                ))

        elif analysis.get("has_main_go"):
            commands.extend([
                BuildCommand("go run main.go", 80, "Run main.go", "run", 85, source="go", requires=["main.go"]),
                BuildCommand("go build main.go", 75, "Build main.go", "build", 80, source="go", requires=["main.go"]),
                BuildCommand("go fmt main.go", 70, "Format main.go", "lint", 70, source="go", requires=["main.go"]),
            ])

        return commands

    def _generate_generic_java_commands(self, analysis: Dict[str, Any]) -> List[BuildCommand]:
        """Generates commands for generic Java projects."""
        return [
            BuildCommand("javac *.java", 60, "Compile Java files", "build", 70, source="java"),
            BuildCommand("java Main", 50, "Run Main class (if exists)", "run", 60, source="java"),
            BuildCommand("jar cf app.jar *.class", 55, "Create JAR file", "build", 65, source="java"),
        ]

    def _validate_tool_availability(self, command: BuildCommand) -> None:
        """Validates if the required tool for the command is available."""
        try:
            main_tool = command.command.split()[0]
            if any(op in command.command for op in ["&&", "||", "|", ">", "<", "cd "]):
                command.validation_result = "complex_command"
                return

            if main_tool == "./gradlew":
                if (self.repo_path / "gradlew").exists():
                    command.validation_result = "valid"
                    command.confidence = min(command.confidence + 10, 98)
                else:
                    command.validation_result = "wrapper_missing"
                    command.confidence = max(command.confidence - 20, 30)
                return

            result = subprocess.run(["which", main_tool], capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                command.validation_result = "tool_available"
                command.confidence = min(command.confidence + 5, 98)
            else:
                command.validation_result = "tool_missing"
                command.confidence = max(command.confidence - 10, 40)
        except subprocess.TimeoutExpired:
            command.validation_result = "validation_timeout"
        except Exception as e:
            command.validation_result = "validation_error"
            logger.debug(f"Validation failed for {command.command}: {e}")

class EnhancedAIProvider:
    """Provides AI-enhanced analysis for build command generation."""

    def __init__(self, provider_type: str, **kwargs):
        self.provider_type = provider_type.lower()
        self.model = kwargs.get("model", self._get_default_model())
        self.api_key = kwargs.get("api_key")
        self.host = kwargs.get("host", "http://localhost:11434")
        if self.provider_type == "ollama":
            self._setup_ollama()

    def _get_default_model(self) -> str:
        """Returns the default AI model for the provider."""
        return {
            "ollama": "gpt-oss:latest",
            "claude": "claude-4-sonnet",
            "openai": "gpt-5",
        }.get(self.provider_type, "gpt-oss:latest")

    def _setup_ollama(self):
        """Sets up and validates Ollama connection."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code != 200:
                raise Exception(f"Ollama not accessible at {self.host}")

            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            if not any(self.model in name for name in model_names):
                logger.warning(f"Model {self.model} not found, attempting to pull...")
                pull_response = requests.post(
                    f"{self.host}/api/pull",
                    json={"name": self.model, "stream": False},
                    timeout=120
                )
                if pull_response.status_code != 200:
                    logger.error(f"Failed to pull model {self.model}")
                    raise Exception(f"Model {self.model} unavailable")
        except Exception as e:
            logger.error(f"Ollama setup failed: {e}")
            raise

    def analyze_repository(self, project_summary: str, key_files: Dict[str, str]) -> List[BuildCommand]:
        """Analyzes repository using AI."""
        if self.provider_type == "ollama":
            return self._analyze_with_ollama(project_summary, key_files)
        logger.error(f"Unsupported AI provider: {self.provider_type}")
        return []

    def _analyze_with_ollama(self, project_summary: str, key_files: Dict[str, str]) -> List[BuildCommand]:
        """Performs Ollama-based analysis."""
        try:
            prompt = f"""Analyze this repository and provide essential build commands:

{project_summary}

Key files:
{self._format_key_files(key_files)}

Return only a JSON array: [{{"command": "npm install", "confidence": 95, "description": "Install dependencies", "category": "install", "priority": 100, "requires": ["package.json"]}}]"""
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "top_k": 10, "top_p": 0.9, "num_ctx": 4096},
                },
                timeout=60,
            )
            if response.status_code == 200:
                result = response.json()
                return self._parse_ai_response(result.get("response", ""))
            logger.warning(f"Ollama request failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"Ollama analysis failed: {e}")
        return []

    def _format_key_files(self, key_files: Dict[str, str]) -> str:
        """Formats key files for AI consumption."""
        formatted = ""
        for file_path, content in key_files.items():
            formatted += f"\n--- {file_path} ---\n"
            formatted += content[:1500] + "\n...[truncated]" if len(content) > 1500 else content
            formatted += "\n"
        return formatted

    def _parse_ai_response(self, response: str) -> List[BuildCommand]:
        """Parses AI response into build commands."""
        commands = []
        try:
            json_matches = re.findall(r"\[[\s\S]*?\]", response)
            for json_text in json_matches:
                json_text = re.sub(r"```json\s*", "", json_text)
                json_text = re.sub(r"```\s*", "", json_text)
                commands_data = json.loads(json_text)
                if isinstance(commands_data, list):
                    for cmd_data in commands_data:
                        if isinstance(cmd_data, dict) and cmd_data.get("command"):
                            commands.append(BuildCommand(
                                command=cmd_data["command"].strip(),
                                confidence=max(30, min(95, cmd_data.get("confidence", 70))),
                                source="ai_enhanced",
                                description=cmd_data.get("description", "AI generated command"),
                                category=cmd_data.get("category", "build"),
                                priority=cmd_data.get("priority", 50),
                                requires=cmd_data.get("requires", []),
                            ))
                    break
        except Exception as e:
            logger.warning(f"Failed to parse AI response: {e}")
        return commands

class EnhancedRepositoryAnalyzer:
    """Performs comprehensive repository analysis."""

    def __init__(self, ai_provider: Optional[EnhancedAIProvider] = None, repo_url: Optional[str] = None):
        self.ai_provider = ai_provider
        self.repo_url = repo_url
        self.parser = SafeFileParser()

    def analyze_repository(self, repo_path: str) -> ProjectInfo:
        """Analyzes the repository and generates build commands."""
        repo_path = Path(repo_path).resolve()
        errors = []

        try:
            detector = ProjectTypeDetector(repo_path)
            detections = detector.detect_projects()
            subprojects = detector.find_subprojects()

            project_type, confidence, detection_data = detections[0] if detections else ("unknown", 0.0, {})
            logger.info(f"Detected project types: {[t[0] for t in detections]}")

            command_generator = SmartCommandGenerator(repo_path)
            all_commands = []
            for typ, conf, data in detections:
                all_commands.extend(command_generator.generate_commands(typ, data))

            for sub_name, sub_type, sub_conf, sub_data in subprojects:
                sub_commands = command_generator.generate_commands(sub_type, sub_data)
                for cmd in sub_commands:
                    cmd.command = f"cd {sub_name} && {cmd.command}"
                    cmd.description += f" (in subproject {sub_name})"
                    cmd.source += f"_{sub_name}"
                all_commands.extend(sub_commands)

            if self.ai_provider:
                try:
                    project_summary = self._create_project_summary(detections, subprojects)
                    key_files = self._extract_key_files(repo_path, [t[0] for t in detections], subprojects)
                    ai_commands = self.ai_provider.analyze_repository(project_summary, key_files)
                    all_commands.extend(ai_commands)
                    logger.info(f"AI enhanced analysis with {len(ai_commands)} additional commands")
                except Exception as e:
                    errors.append(f"AI enhancement failed: {e}")

            processed_commands = self._process_commands(all_commands)

            return ProjectInfo(
                project_type=project_type,
                primary_language=self._map_type_to_language(project_type),
                languages=self._analyze_languages(repo_path),
                build_system=self._determine_build_system(project_type, detection_data),
                package_manager=self._determine_package_manager(project_type, detection_data),
                frameworks=detection_data.get("frameworks", []),
                build_commands=processed_commands,
                analysis_confidence=confidence,
                project_structure=self._analyze_structure(repo_path),
                entry_points=self._find_entry_points(repo_path, project_type),
                errors=errors,
                subprojects=[{"name": n, "type": t, "confidence": c, "data": d} for n, t, c, d in subprojects],
            )

        except Exception as e:
            errors.append(f"Analysis failed: {e}")
            logger.error(f"Analysis failed: {e}")
            return ProjectInfo(
                project_type="unknown",
                primary_language="unknown",
                languages={},
                build_system="unknown",
                package_manager="unknown",
                frameworks=[],
                build_commands=[],
                analysis_confidence=0.0,
                project_structure={},
                entry_points=[],
                errors=errors,
                subprojects=[],
            )

    def _create_project_summary(self, detections: List[Tuple[str, float, Dict]], subprojects: List[Tuple[str, str, float, Dict]]) -> str:
        """Creates a summary for AI analysis."""
        summary = "Detected Project Types:\n"
        for typ, conf, data in detections:
            summary += f"- {typ} (confidence: {conf:.1f}%)\n"
            if data.get("build_system"):
                summary += f"  Build System: {data['build_system']}\n"
            if data.get("package_manager"):
                summary += f"  Package Manager: {data['package_manager']}\n"
            if data.get("frameworks"):
                summary += f"  Frameworks: {', '.join(data['frameworks'])}\n"

        if subprojects:
            summary += "\nSubprojects:\n"
            for name, typ, conf, _ in subprojects:
                summary += f"- {name}: {typ} (confidence: {conf:.1f}%)\n"

        return summary

    def _extract_key_files(self, repo_path: Path, project_types: List[str], subprojects: List[Tuple[str, str, float, Dict]]) -> Dict[str, str]:
        """Extracts key configuration files."""
        key_files = {}
        file_patterns = {
            "nodejs": ["package.json", "tsconfig.json", "webpack.config.js", ".nvmrc"],
            "spring_boot": ["pom.xml", "build.gradle", "application.properties", "application.yml"],
            "java_maven": ["pom.xml"],
            "java_gradle": ["build.gradle", "build.gradle.kts", "settings.gradle"],
            "python": ["requirements.txt", "setup.py", "pyproject.toml", "Pipfile", "poetry.lock"],
            "go": ["go.mod", "go.sum", "main.go"],
            "java": ["pom.xml", "build.gradle"],
        }

        files_to_check = set()
        for typ in project_types:
            files_to_check.update(file_patterns.get(typ, []))
        files_to_check.update(["README.md", "Dockerfile", "docker-compose.yml"])

        for file_name in files_to_check:
            file_path = repo_path / file_name
            if file_path.exists():
                content = self.parser.read_file_safe(file_path, max_size=50_000)
                if content:
                    key_files[str(file_path.relative_to(repo_path))] = content

        for sub_name, sub_type, _, _ in subprojects:
            sub_path = repo_path / sub_name
            for file_name in file_patterns.get(sub_type, []):
                file_path = sub_path / file_name
                if file_path.exists():
                    content = self.parser.read_file_safe(file_path, max_size=50_000)
                    if content:
                        key_files[str(file_path.relative_to(repo_path))] = content

        return key_files

    def _process_commands(self, commands: List[BuildCommand]) -> List[BuildCommand]:
        """Processes and deduplicates commands."""
        seen_commands = {}
        for cmd in commands:
            key = cmd.command.lower().strip()
            if key not in seen_commands or cmd.confidence > seen_commands[key].confidence:
                seen_commands[key] = cmd

        unique_commands = list(seen_commands.values())
        category_priority = {
            "install": 1000, "build": 900, "run": 800, "dev": 750,
            "test": 700, "clean": 600, "lint": 500, "other": 400
        }

        unique_commands.sort(
            key=lambda cmd: (category_priority.get(cmd.category, 400), cmd.priority, cmd.confidence),
            reverse=True
        )
        return unique_commands[:15]

    def _map_type_to_language(self, project_type: str) -> str:
        """Maps project type to primary language."""
        return {
            "nodejs": "javascript", "spring_boot": "java", "java_maven": "java",
            "java_gradle": "java", "java": "java", "python": "python", "go": "go"
        }.get(project_type, "unknown")

    def _determine_build_system(self, project_type: str, detection_data: Dict) -> str:
        """Determines the build system."""
        return detection_data.get("build_system", {
            "nodejs": "npm", "spring_boot": "maven", "java_maven": "maven",
            "java_gradle": "gradle", "python": "pip", "go": "go"
        }.get(project_type, "unknown"))

    def _determine_package_manager(self, project_type: str, detection_data: Dict) -> str:
        """Determines the package manager."""
        return detection_data.get("package_manager", {
            "nodejs": "npm", "spring_boot": "maven", "java_maven": "maven",
            "java_gradle": "gradle", "python": "pip", "go": "go"
        }.get(project_type, "unknown"))

    def _analyze_languages(self, repo_path: Path) -> Dict[str, float]:
        """Analyzes language distribution in the repository."""
        local_bytes = Counter()
        for file_path in repo_path.rglob("*"):
            if file_path.is_file() and not self._should_ignore_file(file_path):
                ext = file_path.suffix.lower()
                lang = self._get_lang_from_ext(ext)
                if lang:
                    try:
                        size = file_path.stat().st_size
                        local_bytes[lang] += size
                    except Exception:
                        pass

        total_local = sum(local_bytes.values())
        local_languages = {lang: (b / total_local * 100) if total_local else 0 for lang, b in local_bytes.items()}

        github_languages = None
        if self.repo_url and "github.com" in self.repo_url:
            try:
                match = re.match(r"https?://github\.com/([^/]+)/([^/]+)", self.repo_url)
                if match:
                    owner, repo_name = match.groups()
                    if repo_name.endswith(".git"):
                        repo_name = repo_name[:-4]
                    api_url = f"https://api.github.com/repos/{owner}/{repo_name}/languages"
                    headers = {"Accept": "application/vnd.github.v3+json"}
                    resp = requests.get(api_url, headers=headers, timeout=5)
                    if resp.status_code == 200:
                        data = resp.json()
                        total_gh = sum(data.values())
                        github_languages = {lang.lower(): (b / total_gh * 100) if total_gh else 0 for lang, b in data.items()}
            except Exception as e:
                logger.debug(f"GitHub API error: {e}")

        if github_languages:
            # Cross-check primaries
            primary_local = max(local_languages, key=local_languages.get, default="unknown")
            primary_gh = max(github_languages, key=github_languages.get, default="unknown")
            if primary_local != primary_gh:
                logger.warning(f"Language primary mismatch: local {primary_local}, GitHub {primary_gh}")

            # Merge by averaging
            all_langs = set(local_languages) | set(github_languages)
            merged = {}
            for lang in all_langs:
                l = local_languages.get(lang, 0)
                g = github_languages.get(lang, 0)
                merged[lang] = (l + g) / 2 if l and g else l or g
            total_merged = sum(merged.values())
            languages = {lang: (p / total_merged * 100) if total_merged else 0 for lang, p in merged.items()}
        else:
            languages = local_languages

        return languages

    def _get_lang_from_ext(self, ext: str) -> Optional[str]:
        """Maps extension to language."""
        ext_lang_map = {
            ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
            ".ts": "typescript", ".tsx": "typescript", ".py": "python",
            ".java": "java", ".go": "go", ".rs": "rust", ".cpp": "cpp",
            ".cc": "cpp", ".cxx": "cpp", ".c": "c", ".cs": "csharp",
            ".rb": "ruby", ".php": "php"
        }
        return ext_lang_map.get(ext)

    def _analyze_structure(self, repo_path: Path) -> Dict[str, Any]:
        """Analyzes the project directory structure."""
        structure = {"root_files": [], "directories": [], "config_files": [], "source_files": []}
        try:
            for item in repo_path.iterdir():
                if item.is_file() and not self._should_ignore_file(item):
                    structure["root_files"].append(item.name)
                elif item.is_dir() and not self._should_ignore_dir(item):
                    structure["directories"].append(item.name)
        except Exception as e:
            logger.debug(f"Error analyzing structure: {e}")
        return structure

    def _find_entry_points(self, repo_path: Path, project_type: str) -> List[str]:
        """Finds potential entry points for the project."""
        entry_patterns = {
            "nodejs": ["index.js", "app.js", "server.js", "main.js"],
            "python": ["main.py", "app.py", "run.py", "server.py", "manage.py"],
            "go": ["main.go"],
            "java": ["Main.java", "Application.java"],
            "spring_boot": ["Application.java", "Main.java"],
        }
        entry_points = []
        for pattern in entry_patterns.get(project_type, []):
            if (repo_path / pattern).exists():
                entry_points.append(pattern)
        return entry_points

    def _should_ignore_file(self, file_path: Path) -> bool:
        """Checks if a file should be ignored."""
        ignore_patterns = {".git", ".svn", ".hg", "__pycache__", ".pytest_cache",
                           "node_modules", ".DS_Store", "Thumbs.db", ".idea", ".vscode"}
        return (file_path.name.startswith(".") and file_path.name not in [".env", ".gitignore"]) or \
            any(pattern in str(file_path) for pattern in ignore_patterns)

    def _should_ignore_dir(self, dir_path: Path) -> bool:
        """Checks if a directory should be ignored."""
        return dir_path.name in {".git", ".svn", "__pycache__", "node_modules",
                                 ".idea", ".vscode", "build", "dist", "target", ".gradle"}

def create_ai_provider(provider_type: str, **kwargs) -> Optional[EnhancedAIProvider]:
    """Creates an AI provider instance."""
    try:
        return EnhancedAIProvider(provider_type, **kwargs)
    except Exception as e:
        logger.error(f"Failed to create {provider_type} provider: {e}")
        return None

def analyze_and_output(args, repo_path: Path, repo_url: Optional[str]):
    """Performs analysis and outputs results."""
    try:
        ai_provider = None
        if args.ai:
            ai_kwargs = {}
            if args.ai_model:
                ai_kwargs["model"] = args.ai_model
            if args.ai_host:
                ai_kwargs["host"] = args.ai_host
            if args.ai_key:
                ai_kwargs["api_key"] = args.ai_key
            ai_provider = create_ai_provider(args.ai, **ai_kwargs)

        analyzer = EnhancedRepositoryAnalyzer(ai_provider, repo_url)
        project_info = analyzer.analyze_repository(str(repo_path))

        filtered_commands = [cmd for cmd in project_info.build_commands if args.category == "all" or cmd.category == args.category]

        if args.output == "json":
            output_data = asdict(project_info)
            output_data["build_commands"] = [asdict(cmd) for cmd in filtered_commands[:args.top_commands]]
            print(json.dumps(output_data, indent=2, default=str))
        elif args.output == "yaml":
            output_data = asdict(project_info)
            output_data["build_commands"] = [asdict(cmd) for cmd in filtered_commands[:args.top_commands]]
            print(yaml.dump(output_data, default_flow_style=False))
        else:
            print(f"\n🔍 Repository Analysis: {repo_path.name}")
            print("=" * 60)
            print(f"📋 Project Type: {project_info.project_type.replace('_', ' ').title()}")
            print(f"💻 Primary Language: {project_info.primary_language.title()}")
            print(f"🔧 Build System: {project_info.build_system.title()}")
            print(f"📦 Package Manager: {project_info.package_manager.title()}")
            print(f"📊 Analysis Confidence: {project_info.analysis_confidence:.1f}%")

            if project_info.frameworks:
                print(f"🚀 Frameworks: {', '.join(project_info.frameworks)}")
            if project_info.entry_points:
                print(f"🎯 Entry Points: {', '.join(project_info.entry_points)}")

            if project_info.subprojects:
                print("\n📂 Subprojects:")
                for sub in project_info.subprojects:
                    print(f"  - {sub['name']}: {sub['type']} ({sub['confidence']:.1f}%)")

            if project_info.languages:
                print(f"\n📈 Language Distribution:")
                for lang, percentage in sorted(project_info.languages.items(), key=lambda x: x[1], reverse=True)[:5]:
                    bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
                    print(f"  {lang.title()}: {percentage:.1f}% [{bar}]")

            if filtered_commands:
                print(f"\n⚡ Build Commands ({args.category.title() if args.category != 'all' else 'All'}):")
                print("-" * 50)
                for i, cmd in enumerate(filtered_commands[:args.top_commands], 1):
                    confidence_bar = "█" * (cmd.confidence // 10) + "░" * (10 - cmd.confidence // 10)
                    category_emoji = {
                        "install": "📥", "build": "🔨", "test": "🧪", "run": "🚀",
                        "dev": "💻", "clean": "🧹", "lint": "✨"
                    }.get(cmd.category, "⚙️")
                    print(f"\n{i}. {cmd.command}")
                    print(f"   {category_emoji} {cmd.category.title()} | Confidence: {cmd.confidence}% [{confidence_bar}]")
                    print(f"   💡 {cmd.description}")
                    if cmd.validation_result == "tool_missing":
                        print(f"   ⚠️ Tool may not be installed")
                    elif cmd.validation_result == "tool_available":
                        print(f"   ✅ Tool available")
                    if cmd.requires:
                        print(f"   📋 Requires: {', '.join(cmd.requires)}")
            else:
                print(f"\n❌ No {args.category} commands found!" if args.category != "all" else "\n❌ No build commands detected!")

            if project_info.errors:
                print(f"\n⚠️ Warnings:")
                for error in project_info.errors:
                    print(f"   • {error}")

            print(f"\n📋 Analysis Summary:")
            print(f"   • Confidence: {project_info.analysis_confidence:.1f}%")
            print(f"   • Commands found: {len(project_info.build_commands)}")
            if project_info.analysis_confidence < 70:
                print("   💡 Consider using --ai flag for enhanced analysis")
            if ai_provider:
                print(f"   🤖 AI-enhanced analysis completed")

    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def main():
    """Main entry point for repository analysis."""
    parser = argparse.ArgumentParser(
        description="Smart Repository Build Command Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyzer.py /path/to/repo
  python analyzer.py https://github.com/owner/repo
  python analyzer.py /path/to/repo --ai ollama --ai-model gpt-oss:latest
  python analyzer.py /path/to/repo --output json
  python analyzer.py /path/to/repo --category build --top-commands 5
        """
    )

    parser.add_argument("repo", help="Path or URL to the repository to analyze")
    parser.add_argument("--output", "-o", choices=["json", "yaml", "text"], default="text", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--top-commands", "-n", type=int, default=10, help="Number of top commands to show")
    parser.add_argument("--category", "-c", choices=["all", "install", "build", "test", "run", "dev", "clean", "lint"],
                        default="all", help="Filter commands by category")
    ai_group = parser.add_argument_group("AI Enhancement Options")
    ai_group.add_argument("--ai", choices=["ollama", "claude", "openai"], help="Enable AI analysis")
    ai_group.add_argument("--ai-model", help="AI model to use")
    ai_group.add_argument("--ai-host", default="http://localhost:11434", help="AI host URL for Ollama")
    ai_group.add_argument("--ai-key", help="API key for Claude/OpenAI")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    is_url = args.repo.startswith(('http://', 'https://'))
    if is_url:
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                subprocess.check_call(["git", "clone", "--depth=1", args.repo, temp_dir])
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone repository from URL {args.repo}: {e}")
                sys.exit(1)
            except FileNotFoundError:
                logger.error("Git is not installed or not in PATH.")
                sys.exit(1)
            repo_path = Path(temp_dir)
            analyze_and_output(args, repo_path, args.repo)
    else:
        repo_path = Path(args.repo).resolve()
        if not repo_path.exists() or not repo_path.is_dir():
            logger.error(f"Invalid repository path: {args.repo}")
            sys.exit(1)
        analyze_and_output(args, repo_path, None)

if __name__ == "__main__":
    main()
