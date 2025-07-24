# Code Quality Agent - Security & Standards Enforcement
# Handles code quality analysis, security scanning, and standards enforcement

import asyncio
import json
import time
import subprocess
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

# LangChain imports
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import BaseTool, tool
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Core framework imports
from core_agent_framework import (
    BaseSDLCAgent, AgentConfiguration, AgentCapability, 
    AgentContext, LLMProvider, AgentState
)

# Tool integrations
import httpx
import requests
from bandit.core import manager as bandit_manager
from bandit.core import config as bandit_config

class StaticCodeAnalysisTool(BaseTool):
    """Tool for comprehensive static code analysis"""
    
    name = "static_code_analysis"
    description = "Perform static code analysis for quality and security issues"
    
    def _run(self, code_files: Dict[str, str], language: str, 
             analysis_rules: List[str] = None) -> Dict:
        """Analyze code for quality and security issues"""
        
        analysis_results = {
            "overall_score": 0,
            "security_score": 0,
            "maintainability_score": 0,
            "reliability_score": 0,
            "files_analyzed": len(code_files),
            "issues": [],
            "metrics": {
                "cyclomatic_complexity": {},
                "code_duplication": 0,
                "technical_debt": 0,
                "test_coverage": 0
            },
            "recommendations": []
        }
        
        total_score = 0
        security_issues = 0
        quality_issues = 0
        
        for file_path, code_content in code_files.items():
            file_analysis = self._analyze_file(file_path, code_content, language)
            
            # Aggregate results
            analysis_results["issues"].extend(file_analysis["issues"])
            total_score += file_analysis["quality_score"]
            
            # Count security issues
            security_issues += len([i for i in file_analysis["issues"] 
                                  if i["category"] == "security"])
            quality_issues += len([i for i in file_analysis["issues"] 
                                 if i["category"] == "quality"])
            
            # Update complexity metrics
            analysis_results["metrics"]["cyclomatic_complexity"][file_path] = \
                file_analysis["complexity"]
        
        # Calculate overall scores
        if code_files:
            analysis_results["overall_score"] = total_score / len(code_files)
            analysis_results["security_score"] = max(0, 100 - (security_issues * 10))
            analysis_results["maintainability_score"] = max(0, 100 - (quality_issues * 5))
            analysis_results["reliability_score"] = analysis_results["overall_score"]
        
        # Calculate technical debt (simplified)
        total_issues = len(analysis_results["issues"])
        analysis_results["metrics"]["technical_debt"] = total_issues * 15  # minutes
        
        # Generate recommendations
        analysis_results["recommendations"] = self._generate_recommendations(
            analysis_results["issues"], language
        )
        
        return analysis_results
    
    def _analyze_file(self, file_path: str, code_content: str, language: str) -> Dict:
        """Analyze individual file for issues"""
        
        issues = []
        complexity = 0
        quality_score = 85  # Base score
        
        lines = code_content.split('\n')
        
        if language.lower() == "python":
            issues.extend(self._analyze_python_file(file_path, code_content, lines))
        elif language.lower() in ["javascript", "typescript"]:
            issues.extend(self._analyze_js_file(file_path, code_content, lines))
        elif language.lower() == "java":
            issues.extend(self._analyze_java_file(file_path, code_content, lines))
        
        # Calculate complexity
        complexity = self._calculate_complexity(code_content, language)
        
        # Adjust quality score based on issues
        quality_score -= len(issues) * 2
        quality_score = max(0, quality_score)
        
        return {
            "file_path": file_path,
            "issues": issues,
            "complexity": complexity,
            "quality_score": quality_score,
            "lines_of_code": len(lines)
        }
    
    def _analyze_python_file(self, file_path: str, code_content: str, lines: List[str]) -> List[Dict]:
        """Analyze Python file for specific issues"""
        issues = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Security issues
            if "eval(" in line_stripped:
                issues.append({
                    "file": file_path,
                    "line": i + 1,
                    "severity": "high",
                    "category": "security",
                    "rule": "no-eval",
                    "message": "Use of eval() is dangerous and should be avoided",
                    "recommendation": "Use safer alternatives like ast.literal_eval()"
                })
            
            if "exec(" in line_stripped:
                issues.append({
                    "file": file_path,
                    "line": i + 1,
                    "severity": "high",
                    "category": "security",
                    "rule": "no-exec",
                    "message": "Use of exec() is dangerous",
                    "recommendation": "Avoid dynamic code execution"
                })
            
            if "pickle.loads(" in line_stripped:
                issues.append({
                    "file": file_path,
                    "line": i + 1,
                    "severity": "medium",
                    "category": "security",
                    "rule": "unsafe-deserialization",
                    "message": "Unsafe deserialization with pickle",
                    "recommendation": "Use safer serialization formats like JSON"
                })
            
            # Code quality issues
            if len(line) > 120:
                issues.append({
                    "file": file_path,
                    "line": i + 1,
                    "severity": "minor",
                    "category": "quality",
                    "rule": "line-too-long",
                    "message": f"Line too long ({len(line)} > 120 characters)",
                    "recommendation": "Break long lines for better readability"
                })
            
            if line_stripped.startswith('def ') and not any('"""' in l or "'''" in l for l in lines[i:i+5]):
                issues.append({
                    "file": file_path,
                    "line": i + 1,
                    "severity": "minor",
                    "category": "quality",
                    "rule": "missing-docstring",
                    "message": "Function missing docstring",
                    "recommendation": "Add docstring to document function purpose"
                })
            
            # Detect hardcoded secrets
            secret_patterns = [
                r"password\s*=\s*['\"][^'\"]+['\"]",
                r"api_key\s*=\s*['\"][^'\"]+['\"]",
                r"secret\s*=\s*['\"][^'\"]+['\"]",
                r"token\s*=\s*['\"][^'\"]+['\"]"
            ]
            
            for pattern in secret_patterns:
                if re.search(pattern, line_stripped, re.IGNORECASE):
                    issues.append({
                        "file": file_path,
                        "line": i + 1,
                        "severity": "high",
                        "category": "security",
                        "rule": "hardcoded-secrets",
                        "message": "Hardcoded secret detected",
                        "recommendation": "Use environment variables or secure vaults"
                    })
        
        return issues
    
    def _analyze_js_file(self, file_path: str, code_content: str, lines: List[str]) -> List[Dict]:
        """Analyze JavaScript/TypeScript file for issues"""
        issues = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Security issues
            if "eval(" in line_stripped:
                issues.append({
                    "file": file_path,
                    "line": i + 1,
                    "severity": "high",
                    "category": "security",
                    "rule": "no-eval",
                    "message": "Use of eval() is dangerous",
                    "recommendation": "Use safer alternatives"
                })
            
            if "innerHTML" in line_stripped and "=" in line_stripped:
                issues.append({
                    "file": file_path,
                    "line": i + 1,
                    "severity": "medium",
                    "category": "security",
                    "rule": "xss-risk",
                    "message": "Potential XSS vulnerability with innerHTML",
                    "recommendation": "Use textContent or sanitize input"
                })
            
            # Code quality issues
            if "console.log(" in line_stripped:
                issues.append({
                    "file": file_path,
                    "line": i + 1,
                    "severity": "minor",
                    "category": "quality",
                    "rule": "no-console",
                    "message": "Console statement found",
                    "recommendation": "Remove console statements in production"
                })
            
            if "var " in line_stripped:
                issues.append({
                    "file": file_path,
                    "line": i + 1,
                    "severity": "minor",
                    "category": "quality",
                    "rule": "no-var",
                    "message": "Use let or const instead of var",
                    "recommendation": "Replace var with let or const"
                })
        
        return issues
    
    def _analyze_java_file(self, file_path: str, code_content: str, lines: List[str]) -> List[Dict]:
        """Analyze Java file for issues"""
        issues = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Security issues
            if "Runtime.getRuntime().exec(" in line_stripped:
                issues.append({
                    "file": file_path,
                    "line": i + 1,
                    "severity": "high",
                    "category": "security",
                    "rule": "command-injection",
                    "message": "Potential command injection vulnerability",
                    "recommendation": "Validate and sanitize command inputs"
                })
            
            if "System.out.println(" in line_stripped:
                issues.append({
                    "file": file_path,
                    "line": i + 1,
                    "severity": "minor",
                    "category": "quality",
                    "rule": "no-system-out",
                    "message": "Use logging framework instead of System.out",
                    "recommendation": "Replace with logger.info() or similar"
                })
        
        return issues
    
    def _calculate_complexity(self, code_content: str, language: str) -> int:
        """Calculate cyclomatic complexity (simplified)"""
        complexity = 1  # Base complexity
        
        # Count decision points
        decision_keywords = {
            "python": ["if", "elif", "while", "for", "try", "except", "and", "or"],
            "javascript": ["if", "else if", "while", "for", "try", "catch", "&&", "||"],
            "java": ["if", "else if", "while", "for", "try", "catch", "&&", "||"]
        }
        
        keywords = decision_keywords.get(language.lower(), decision_keywords["python"])
        
        for keyword in keywords:
            complexity += code_content.count(keyword)
        
        return min(complexity, 50)  # Cap at 50
    
    def _generate_recommendations(self, issues: List[Dict], language: str) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        security_issues = [i for i in issues if i["category"] == "security"]
        quality_issues = [i for i in issues if i["category"] == "quality"]
        
        if security_issues:
            recommendations.append(
                f"Address {len(security_issues)} security vulnerabilities immediately"
            )
            recommendations.append("Implement security scanning in CI/CD pipeline")
        
        if quality_issues:
            recommendations.append(
                f"Fix {len(quality_issues)} code quality issues"
            )
            recommendations.append("Set up automated code formatting")
        
        # Language-specific recommendations
        if language.lower() == "python":
            recommendations.extend([
                "Use tools like bandit for security scanning",
                "Implement type hints for better code clarity",
                "Use black for code formatting"
            ])
        elif language.lower() in ["javascript", "typescript"]:
            recommendations.extend([
                "Use ESLint for code quality checks",
                "Implement Prettier for code formatting",
                "Consider TypeScript for better type safety"
            ])
        
        return recommendations[:5]  # Limit to top 5
    
    async def _arun(self, code_files: Dict[str, str], language: str, 
                   analysis_rules: List[str] = None) -> Dict:
        """Async version"""
        return self._run(code_files, language, analysis_rules)

class SecurityScannerTool(BaseTool):
    """Tool for security vulnerability scanning"""
    
    name = "security_scanner"
    description = "Scan code for security vulnerabilities and compliance issues"
    
    def _run(self, code_files: Dict[str, str], scan_type: str = "comprehensive") -> Dict:
        """Perform security scanning"""
        
        scan_results = {
            "scan_type": scan_type,
            "total_files": len(code_files),
            "vulnerabilities": [],
            "compliance_issues": [],
            "security_score": 0,
            "risk_level": "low",
            "recommendations": []
        }
        
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0
        
        for file_path, code_content in code_files.items():
            file_vulnerabilities = self._scan_file_security(file_path, code_content)
            scan_results["vulnerabilities"].extend(file_vulnerabilities)
            
            # Count by severity
            for vuln in file_vulnerabilities:
                if vuln["severity"] == "high":
                    high_risk_count += 1
                elif vuln["severity"] == "medium":
                    medium_risk_count += 1
                else:
                    low_risk_count += 1
        
        # Calculate security score
        total_vulns = high_risk_count + medium_risk_count + low_risk_count
        if total_vulns == 0:
            scan_results["security_score"] = 100
            scan_results["risk_level"] = "low"
        else:
            # Weighted scoring
            weighted_score = (high_risk_count * 20) + (medium_risk_count * 10) + (low_risk_count * 5)
            scan_results["security_score"] = max(0, 100 - weighted_score)
            
            if high_risk_count > 0:
                scan_results["risk_level"] = "high"
            elif medium_risk_count > 2:
                scan_results["risk_level"] = "medium"
            else:
                scan_results["risk_level"] = "low"
        
        # Generate recommendations
        scan_results["recommendations"] = self._generate_security_recommendations(
            scan_results["vulnerabilities"]
        )
        
        # Check compliance
        scan_results["compliance_issues"] = self._check_compliance(code_files)
        
        return scan_results
    
    def _scan_file_security(self, file_path: str, code_content: str) -> List[Dict]:
        """Scan individual file for security vulnerabilities"""
        vulnerabilities = []
        lines = code_content.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # SQL Injection patterns
            sql_patterns = [
                r"execute\s*\(\s*['\"].*\+.*['\"]",
                r"query\s*\(\s*['\"].*\+.*['\"]",
                r"SELECT.*\+.*FROM",
                r"INSERT.*\+.*VALUES"
            ]
            
            for pattern in sql_patterns:
                if re.search(pattern, line_stripped, re.IGNORECASE):
                    vulnerabilities.append({
                        "file": file_path,
                        "line": i + 1,
                        "severity": "high",
                        "type": "sql_injection",
                        "cwe": "CWE-89",
                        "message": "Potential SQL injection vulnerability",
                        "recommendation": "Use parameterized queries",
                        "owasp_category": "A03:2021 – Injection"
                    })
            
            # XSS patterns
            if re.search(r"innerHTML\s*=\s*.*\+", line_stripped):
                vulnerabilities.append({
                    "file": file_path,
                    "line": i + 1,
                    "severity": "medium",
                    "type": "xss",
                    "cwe": "CWE-79",
                    "message": "Potential Cross-Site Scripting (XSS) vulnerability",
                    "recommendation": "Sanitize user input and use textContent",
                    "owasp_category": "A03:2021 – Injection"
                })
            
            # Path traversal
            if re.search(r"open\s*\(\s*.*\+.*\)", line_stripped):
                vulnerabilities.append({
                    "file": file_path,
                    "line": i + 1,
                    "severity": "high",
                    "type": "path_traversal",
                    "cwe": "CWE-22",
                    "message": "Potential path traversal vulnerability",
                    "recommendation": "Validate and sanitize file paths",
                    "owasp_category": "A01:2021 – Broken Access Control"
                })
            
            # Weak cryptography
            weak_crypto_patterns = [
                r"md5\(",
                r"sha1\(",
                r"DES\(",
                r"RC4\("
            ]
            
            for pattern in weak_crypto_patterns:
                if re.search(pattern, line_stripped, re.IGNORECASE):
                    vulnerabilities.append({
                        "file": file_path,
                        "line": i + 1,
                        "severity": "medium",
                        "type": "weak_cryptography",
                        "cwe": "CWE-327",
                        "message": "Use of weak cryptographic algorithm",
                        "recommendation": "Use stronger algorithms like SHA-256 or AES",
                        "owasp_category": "A02:2021 – Cryptographic Failures"
                    })
            
            # Insecure random number generation
            if re.search(r"random\(\)|Math\.random\(\)", line_stripped):
                vulnerabilities.append({
                    "file": file_path,
                    "line": i + 1,
                    "severity": "low",
                    "type": "weak_randomness",
                    "cwe": "CWE-330",
                    "message": "Use of cryptographically weak random number generator",
                    "recommendation": "Use cryptographically secure random generators",
                    "owasp_category": "A02:2021 – Cryptographic Failures"
                })
        
        return vulnerabilities
    
    def _check_compliance(self, code_files: Dict[str, str]) -> List[Dict]:
        """Check compliance with security standards"""
        compliance_issues = []
        
        # Check for missing security headers
        for file_path, code_content in code_files.items():
            if "app.py" in file_path or "main.py" in file_path:
                if "Content-Security-Policy" not in code_content:
                    compliance_issues.append({
                        "file": file_path,
                        "standard": "OWASP",
                        "requirement": "Security Headers",
                        "issue": "Missing Content-Security-Policy header",
                        "severity": "medium",
                        "recommendation": "Implement CSP headers"
                    })
                
                if "X-Frame-Options" not in code_content:
                    compliance_issues.append({
                        "file": file_path,
                        "standard": "OWASP",
                        "requirement": "Clickjacking Protection",
                        "issue": "Missing X-Frame-Options header",
                        "severity": "low",
                        "recommendation": "Add X-Frame-Options: DENY header"
                    })
        
        return compliance_issues
    
    def _generate_security_recommendations(self, vulnerabilities: List[Dict]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        vuln_types = {}
        for vuln in vulnerabilities:
            vuln_type = vuln["type"]
            if vuln_type not in vuln_types:
                vuln_types[vuln_type] = 0
            vuln_types[vuln_type] += 1
        
        if "sql_injection" in vuln_types:
            recommendations.append("Implement parameterized queries to prevent SQL injection")
        
        if "xss" in vuln_types:
            recommendations.append("Implement input sanitization and output encoding")
        
        if "weak_cryptography" in vuln_types:
            recommendations.append("Upgrade to stronger cryptographic algorithms")
        
        recommendations.extend([
            "Implement security scanning in CI/CD pipeline",
            "Regular security training for development team",
            "Conduct periodic penetration testing",
            "Implement Web Application Firewall (WAF)"
        ])
        
        return recommendations[:5]
    
    async def _arun(self, code_files: Dict[str, str], scan_type: str = "comprehensive") -> Dict:
        """Async version"""
        return self._run(code_files, scan_type)

class ComplianceCheckerTool(BaseTool):
    """Tool for checking coding standards and compliance"""
    
    name = "compliance_checker"
    description = "Check code compliance with standards and best practices"
    
    def _run(self, code_files: Dict[str, str], standards: List[str] = None) -> Dict:
        """Check compliance with coding standards"""
        
        standards = standards or ["pep8", "owasp", "sonarqube"]
        
        compliance_results = {
            "standards_checked": standards,
            "overall_compliance": 0,
            "violations": [],
            "compliance_by_standard": {},
            "recommendations": []
        }
        
        total_violations = 0
        
        for standard in standards:
            standard_violations = self._check_standard_compliance(code_files, standard)
            compliance_results["violations"].extend(standard_violations)
            compliance_results["compliance_by_standard"][standard] = {
                "violations": len(standard_violations),
                "compliance_percentage": max(0, 100 - len(standard_violations) * 5)
            }
            total_violations += len(standard_violations)
        
        # Calculate overall compliance
        compliance_results["overall_compliance"] = max(0, 100 - (total_violations * 2))
        
        # Generate recommendations
        compliance_results["recommendations"] = self._generate_compliance_recommendations(
            compliance_results["violations"]
        )
        
        return compliance_results
    
    def _check_standard_compliance(self, code_files: Dict[str, str], standard: str) -> List[Dict]:
        """Check compliance with specific standard"""
        violations = []
        
        if standard == "pep8":
            violations.extend(self._check_pep8_compliance(code_files))
        elif standard == "owasp":
            violations.extend(self._check_owasp_compliance(code_files))
        elif standard == "sonarqube":
            violations.extend(self._check_sonarqube_rules(code_files))
        
        return violations
    
    def _check_pep8_compliance(self, code_files: Dict[str, str]) -> List[Dict]:
        """Check PEP 8 compliance for Python files"""
        violations = []
        
        for file_path, code_content in code_files.items():
            if not file_path.endswith('.py'):
                continue
                
            lines = code_content.split('\n')
            
            for i, line in enumerate(lines):
                # Line length check
                if len(line) > 79:
                    violations.append({
                        "file": file_path,
                        "line": i + 1,
                        "standard": "PEP 8",
                        "rule": "E501",
                        "message": f"Line too long ({len(line)} > 79 characters)",
                        "severity": "minor"
                    })
                
                # Whitespace issues
                if line.endswith(' ') or line.endswith('\t'):
                    violations.append({
                        "file": file_path,
                        "line": i + 1,
                        "standard": "PEP 8",
                        "rule": "W291",
                        "message": "Trailing whitespace",
                        "severity": "minor"
                    })
                
                # Import style
                if line.strip().startswith('from ') and ' import *' in line:
                    violations.append({
                        "file": file_path,
                        "line": i + 1,
                        "standard": "PEP 8",
                        "rule": "F403",
                        "message": "Avoid wildcard imports",
                        "severity": "minor"
                    })
        
        return violations
    
    def _check_owasp_compliance(self, code_files: Dict[str, str]) -> List[Dict]:
        """Check OWASP compliance"""
        violations = []
        
        for file_path, code_content in code_files.items():
            # Check for OWASP Top 10 issues
            if "password" in code_content.lower() and "=" in code_content:
                violations.append({
                    "file": file_path,
                    "line": 0,  # General file issue
                    "standard": "OWASP",
                    "rule": "A07:2021",
                    "message": "Potential hardcoded credentials",
                    "severity": "high"
                })
            
            if "admin" in code_content.lower() and "password" in code_content.lower():
                violations.append({
                    "file": file_path,
                    "line": 0,
                    "standard": "OWASP",
                    "rule": "A07:2021",
                    "message": "Default admin credentials detected",
                    "severity": "high"
                })
        
        return violations
    
    def _check_sonarqube_rules(self, code_files: Dict[str, str]) -> List[Dict]:
        """Check SonarQube quality rules"""
        violations = []
        
        for file_path, code_content in code_files.items():
            lines = code_content.split('\n')
            
            # Cognitive complexity
            complexity = self._calculate_cognitive_complexity(code_content)
            if complexity > 15:
                violations.append({
                    "file": file_path,
                    "line": 1,
                    "standard": "SonarQube",
                    "rule": "S3776",
                    "message": f"Cognitive complexity too high ({complexity} > 15)",
                    "severity": "major"
                })
            
            # TODO comments
            for i, line in enumerate(lines):
                if "TODO" in line.upper() or "FIXME" in line.upper():
                    violations.append({
                        "file": file_path,
                        "line": i + 1,
                        "standard": "SonarQube",
                        "rule": "S1135",
                        "message": "Complete the task associated with this TODO comment",
                        "severity": "info"
                    })
        
        return violations
    
    def _calculate_cognitive_complexity(self, code_content: str) -> int:
        """Calculate cognitive complexity (simplified)"""
        complexity = 0
        
        # Basic increment for control structures
        control_structures = ["if", "else", "elif", "while", "for", "try", "except", "match", "case"]
        
        for structure in control_structures:
            complexity += code_content.count(structure)
        
        return complexity
    
    def _generate_compliance_recommendations(self, violations: List[Dict]) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        standards_with_violations = set(v["standard"] for v in violations)
        
        if "PEP 8" in standards_with_violations:
            recommendations.append("Use automated formatters like black and flake8")
        
        if "OWASP" in standards_with_violations:
            recommendations.append("Implement security scanning tools")
            recommendations.append("Regular security training for developers")
        
        if "SonarQube" in standards_with_violations:
            recommendations.append("Integrate SonarQube into CI/CD pipeline")
            recommendations.append("Address technical debt regularly")
        
        recommendations.extend([
            "Set up pre-commit hooks for code quality checks",
            "Regular code reviews with quality focus",
            "Automated quality gates in deployment pipeline"
        ])
        
        return recommendations[:5]
    
    async def _arun(self, code_files: Dict[str, str], standards: List[str] = None) -> Dict:
        """Async version"""
        return self._run(code_files, standards)

class CodeQualityAgent(BaseSDLCAgent):
    """Code quality agent for security and standards enforcement"""
    
    def __init__(self, config: AgentConfiguration):
        # Define capabilities
        capabilities = [
            AgentCapability(
                name="analyze_code_quality",
                description="Comprehensive code quality and security analysis",
                input_schema={
                    "type": "object",
                    "properties": {
                        "code_repository": {"type": "object"},
                        "quality_standards": {"type": "array"},
                        "security_requirements": {"type": "object"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "quality_report": {"type": "object"},
                        "security_report": {"type": "object"},
                        "compliance_report": {"type": "object"}
                    }
                },
                tools=["static_code_analysis", "security_scanner", "compliance_checker"]
            ),
            AgentCapability(
                name="enforce_quality_gates",
                description="Enforce quality gates and standards",
                input_schema={
                    "type": "object",
                    "properties": {
                        "quality_metrics": {"type": "object"},
                        "gate_thresholds": {"type": "object"},
                        "enforcement_rules": {"type": "array"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "gate_results": {"type": "object"},
                        "violations": {"type": "array"},
                        "recommendations": {"type": "array"}
                    }
                },
                tools=["static_code_analysis", "compliance_checker"]
            ),
            AgentCapability(
                name="continuous_monitoring",
                description="Continuous code quality monitoring",
                input_schema={
                    "type": "object",
                    "properties": {
                        "monitoring_config": {"type": "object"},
                        "alert_thresholds": {"type": "object"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "monitoring_status": {"type": "object"},
                        "alerts": {"type": "array"},
                        "trends": {"type": "object"}
                    }
                },
                tools=["static_code_analysis", "security_scanner"]
            )
        ]
        
        super().__init__(config, capabilities)
        
        # Initialize specialized tools
        self.tools = self._initialize_tools()
        
        # Create LangChain agent
        self.langchain_agent = self._create_langchain_agent()
        
    def _initialize_tools(self) -> List[BaseTool]:
        """Initialize specialized tools for code quality agent"""
        tools = [
            StaticCodeAnalysisTool(),
            SecurityScannerTool(),
            ComplianceCheckerTool()
        ]
        
        return tools
    
    def _create_langchain_agent(self) -> AgentExecutor:
        """Create LangChain agent with specialized prompt"""
        
        system_prompt = """You are a specialized Code Quality Agent for software development lifecycle management.
        
        Your primary responsibilities:
        1. Perform comprehensive code quality analysis
        2. Conduct security vulnerability scanning
        3. Enforce coding standards and compliance
        4. Generate detailed quality reports
        5. Implement quality gates and thresholds
        6. Provide actionable improvement recommendations
        7. Monitor code quality trends over time
        
        Available tools: {tool_names}
        
        When analyzing code quality:
        - Focus on security, maintainability, and reliability
        - Apply industry standards (OWASP, PEP 8, SonarQube rules)
        - Identify critical security vulnerabilities first
        - Provide specific, actionable recommendations
        - Consider code complexity and technical debt
        - Ensure compliance with organizational standards
        
        Always prioritize security issues and provide clear remediation guidance.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_structured_chat_agent(
            llm=self.llm_manager.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )
    
    async def reason(self, input_data: Dict) -> Dict:
        """Reasoning phase: Analyze quality requirements"""
        self.log_execution("reasoning_start", {"input": input_data})
        
        reasoning_prompt = f"""
        Analyze the following code quality task:
        
        Task: {json.dumps(input_data, indent=2)}
        Code Context: {json.dumps(self.context.shared_memory.get('code_context', {}), indent=2)}
        
        Provide analysis covering:
        1. Code quality assessment scope and priorities
        2. Security scanning requirements and threat model
        3. Compliance standards and regulatory requirements
        4. Quality gate thresholds and enforcement rules
        5. Risk assessment for security and quality issues
        6. Technical debt analysis and prioritization
        7. Automation strategy for continuous monitoring
        8. Integration with development workflow
        9. Reporting and communication requirements
        10. Success metrics and improvement tracking
        
        Consider:
        - Application criticality and security requirements
        - Development team practices and maturity
        - Regulatory compliance needs
        - Performance and scalability impact
        - Maintenance and technical debt concerns
        
        Provide structured reasoning with quality strategy recommendations.
        """
        
        reasoning_response = await self.llm_manager.llm.ainvoke([
            HumanMessage(content=reasoning_prompt)
        ])
        
        reasoning_result = {
            "task_understanding": "Comprehensive code quality and security analysis",
            "complexity_assessment": "high",
            "quality_strategy": {
                "primary_focus": "security_first_quality_gates",
                "analysis_depth": "comprehensive",
                "automation_level": "fully_automated",
                "compliance_requirements": ["owasp", "pep8", "sonarqube"]
            },
            "security_priorities": [
                "vulnerability_scanning",
                "dependency_analysis", 
                "secrets_detection",
                "compliance_checking"
            ],
            "quality_thresholds": {
                "minimum_quality_score": 80,
                "maximum_security_violations": 0,
                "code_coverage_requirement": "85_percent",
                "complexity_limit": 15
            },
            "risk_assessment": {
                "security_risk": "high_impact_low_tolerance",
                "quality_risk": "medium_impact_managed_debt",
                "compliance_risk": "regulatory_requirements_strict"
            },
            "automation_strategy": {
                "ci_cd_integration": "mandatory_gates",
                "real_time_monitoring": "enabled",
                "automated_remediation": "where_possible"
            },
            "success_criteria": [
                "zero_high_security_vulnerabilities",
                "quality_gates_passed",
                "compliance_standards_met"
            ],
            "confidence_score": 0.91,
            "reasoning_text": reasoning_response.content
        }
        
        self.log_execution("reasoning_complete", reasoning_result)
        return reasoning_result
    
    async def plan(self, reasoning_output: Dict) -> Dict:
        """Planning phase: Create quality analysis plan"""
        self.log_execution("planning_start", {"reasoning": reasoning_output})
        
        planning_prompt = f"""
        Based on this quality analysis: {json.dumps(reasoning_output, indent=2)}
        
        Create a detailed execution plan including:
        
        1. Static Code Analysis:
           - Code quality scanning across all files
           - Complexity analysis and technical debt assessment
           - Best practices compliance checking
           - Performance and maintainability analysis
        
        2. Security Vulnerability Scanning:
           - OWASP Top 10 vulnerability detection
           - Dependency vulnerability scanning
           - Secrets and credential detection
           - Compliance with security standards
        
        3. Standards Compliance Verification:
           - Coding standards enforcement
           - Regulatory compliance checking
           - Industry best practices validation
           - Custom rule compliance
        
        4. Quality Gate Implementation:
           - Threshold definition and enforcement
           - Automated gate execution
           - Exception handling and approvals
           - Reporting and notifications
        
        5. Continuous Monitoring Setup:
           - Real-time quality monitoring
           - Trend analysis and alerting
           - Dashboard and reporting setup
           - Integration with development tools
        
        Provide specific steps with quality metrics and success criteria.
        """
        
        planning_response = await asyncio.get_event_loop().run_in_executor(
            None, 
            self.langchain_agent.invoke,
            {"input": planning_prompt, "chat_history": []}
        )
        
        plan = {
            "plan_id": f"quality_plan_{int(time.time())}",
            "approach": "comprehensive_quality_assurance",
            "phases": [
                {
                    "phase": "static_analysis",
                    "duration_hours": 6,
                    "steps": [
                        "scan_code_quality",
                        "analyze_complexity_metrics",
                        "detect_code_smells",
                        "assess_technical_debt"
                    ]
                },
                {
                    "phase": "security_scanning",
                    "duration_hours": 8,
                    "steps": [
                        "vulnerability_scanning",
                        "dependency_analysis",
                        "secrets_detection",
                        "security_compliance_check"
                    ]
                },
                {
                    "phase": "compliance_verification",
                    "duration_hours": 4,
                    "steps": [
                        "standards_compliance_check",
                        "regulatory_compliance_verify",
                        "custom_rules_validation",
                        "documentation_compliance"
                    ]
                },
                {
                    "phase": "quality_gates_setup",
                    "duration_hours": 3,
                    "steps": [
                        "define_quality_thresholds",
                        "implement_automated_gates",
                        "setup_exception_handling",
                        "configure_notifications"
                    ]
                },
                {
                    "phase": "monitoring_implementation",
                    "duration_hours": 4,
                    "steps": [
                        "setup_continuous_monitoring",
                        "configure_alerting",
                        "create_quality_dashboards",
                        "integrate_with_tools"
                    ]
                }
            ],
            "tools_to_use": ["static_code_analysis", "security_scanner", "compliance_checker"],
            "deliverables": [
                "comprehensive_quality_report",
                "security_vulnerability_report",
                "compliance_assessment_report",
                "quality_gates_configuration",
                "monitoring_dashboard"
            ],
            "success_metrics": {
                "quality_score": "minimum_80",
                "security_vulnerabilities": "zero_high_critical",
                "compliance_rate": "100_percent"
            },
            "estimated_total_hours": 25,
            "planning_response": planning_response["output"]
        }
        
        self.log_execution("planning_complete", plan)
        return plan
    
    async def act(self, plan: Dict) -> Dict:
        """Action phase: Execute quality analysis plan"""
        self.log_execution("acting_start", {"plan": plan})
        
        results = {
            "execution_id": f"quality_exec_{int(time.time())}",
            "plan_id": plan["plan_id"],
            "phase_results": {},
            "overall_metrics": {},
            "deliverables_created": [],
            "issues_encountered": []
        }
        
        try:
            for phase in plan["phases"]:
                phase_name = phase["phase"]
                self.log_execution(f"phase_start_{phase_name}", phase)
                
                phase_result = await self._execute_phase(phase, plan)
                results["phase_results"][phase_name] = phase_result
                
                self.log_execution(f"phase_complete_{phase_name}", phase_result)
            
            results["overall_metrics"] = await self._compile_metrics(results)
            results["success"] = True
            
        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            self.log_execution("acting_error", {"error": str(e)})
            
        self.log_execution("acting_complete", results)
        return results
    
    async def _execute_phase(self, phase: Dict, overall_plan: Dict) -> Dict:
        """Execute a specific phase of the quality plan"""
        phase_name = phase["phase"]
        
        if phase_name == "static_analysis":
            return await self._execute_static_analysis()
        elif phase_name == "security_scanning":
            return await self._execute_security_scanning()
        elif phase_name == "compliance_verification":
            return await self._execute_compliance_verification()
        elif phase_name == "quality_gates_setup":
            return await self._execute_quality_gates_setup()
        elif phase_name == "monitoring_implementation":
            return await self._execute_monitoring_implementation()
        else:
            return {"status": "not_implemented", "phase": phase_name}
    
    async def _execute_static_analysis(self) -> Dict:
        """Execute static code analysis"""
        analysis_tool = next((tool for tool in self.tools if tool.name == "static_code_analysis"), None)
        
        # Mock code files for analysis
        code_files = {
            "main.py": '''
import os
import hashlib
from flask import Flask, request

app = Flask(__name__)

def authenticate_user(username, password):
    # This is a very long line that exceeds the recommended line length limit and should be flagged by quality tools
    if username == "admin" and password == "password123":  # Hardcoded credentials
        return True
    return False

@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]
    
    # SQL injection vulnerability
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    
    if authenticate_user(username, password):
        return "Success"
    return "Failed"

def weak_hash(data):
    return hashlib.md5(data.encode()).hexdigest()  # Weak cryptography

# TODO: Implement proper error handling
''',
            "utils.js": '''
function processUserInput(input) {
    // XSS vulnerability
    document.getElementById("output").innerHTML = input;
    
    // Use of eval
    var result = eval("2 + 2");
    console.log("Debug: " + result);  // Console statement
    
    var data = input;  // Use of var instead of let/const
    return data;
}

function generateRandomId() {
    return Math.random().toString(36);  // Weak randomness
}
'''
        }
        
        analysis_result = await analysis_tool._arun(
            code_files=code_files,
            language="python",
            analysis_rules=["security", "quality", "complexity"]
        )
        
        return {
            "static_analysis_completed": True,
            "files_analyzed": analysis_result["files_analyzed"],
            "overall_quality_score": analysis_result["overall_score"],
            "security_score": analysis_result["security_score"],
            "maintainability_score": analysis_result["maintainability_score"],
            "issues_found": len(analysis_result["issues"]),
            "technical_debt_minutes": analysis_result["metrics"]["technical_debt"],
            "analysis_details": analysis_result
        }
    
    async def _execute_security_scanning(self) -> Dict:
        """Execute security vulnerability scanning"""
        security_tool = next((tool for tool in self.tools if tool.name == "security_scanner"), None)
        
        # Use same code files for security scanning
        code_files = {
            "main.py": '''
import os
import hashlib
from flask import Flask, request

app = Flask(__name__)

def authenticate_user(username, password):
    if username == "admin" and password == "password123":  # Hardcoded credentials
        return True
    return False

@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]
    
    # SQL injection vulnerability
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    
    if authenticate_user(username, password):
        return "Success"
    return "Failed"

def weak_hash(data):
    return hashlib.md5(data.encode()).hexdigest()  # Weak cryptography
'''
        }
        
        security_result = await security_tool._arun(
            code_files=code_files,
            scan_type="comprehensive"
        )
        
        return {
            "security_scan_completed": True,
            "total_vulnerabilities": len(security_result["vulnerabilities"]),
            "high_risk_vulnerabilities": len([v for v in security_result["vulnerabilities"] if v["severity"] == "high"]),
            "medium_risk_vulnerabilities": len([v for v in security_result["vulnerabilities"] if v["severity"] == "medium"]),
            "low_risk_vulnerabilities": len([v for v in security_result["vulnerabilities"] if v["severity"] == "low"]),
            "security_score": security_result["security_score"],
            "risk_level": security_result["risk_level"],
            "compliance_issues": len(security_result["compliance_issues"]),
            "security_details": security_result
        }
    
    async def _execute_compliance_verification(self) -> Dict:
        """Execute compliance verification"""
        compliance_tool = next((tool for tool in self.tools if tool.name == "compliance_checker"), None)
        
        # Use same code files for compliance checking
        code_files = {
            "main.py": '''
import os
import hashlib
from flask import Flask, request

app = Flask(__name__)

def authenticate_user(username, password):
    # This is a very long line that exceeds the recommended line length limit and should be flagged by quality tools for PEP 8 compliance issues
    if username == "admin" and password == "password123":  # Hardcoded credentials
        return True
    return False

@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]  
    
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    
    if authenticate_user(username, password):
        return "Success"
    return "Failed"

# TODO: Implement proper error handling
# FIXME: Address security vulnerabilities
'''
        }
        
        compliance_result = await compliance_tool._arun(
            code_files=code_files,
            standards=["pep8", "owasp", "sonarqube"]
        )
        
        return {
            "compliance_verification_completed": True,
            "overall_compliance_score": compliance_result["overall_compliance"],
            "standards_checked": len(compliance_result["standards_checked"]),
            "total_violations": len(compliance_result["violations"]),
            "compliance_by_standard": compliance_result["compliance_by_standard"],
            "recommendations_generated": len(compliance_result["recommendations"]),
            "compliance_details": compliance_result
        }
    
    async def _execute_quality_gates_setup(self) -> Dict:
        """Execute quality gates setup"""
        
        # Define quality gates based on analysis results
        quality_gates = {
            "security_gate": {
                "threshold": "zero_high_vulnerabilities",
                "status": "configured",
                "enforcement": "blocking"
            },
            "quality_gate": {
                "threshold": "minimum_80_score",
                "status": "configured", 
                "enforcement": "blocking"
            },
            "compliance_gate": {
                "threshold": "minimum_85_compliance",
                "status": "configured",
                "enforcement": "warning"
            },
            "complexity_gate": {
                "threshold": "maximum_15_complexity",
                "status": "configured",
                "enforcement": "warning"
            }
        }
        
        # Configure automated enforcement
        enforcement_rules = [
            "Block deployment if high security vulnerabilities found",
            "Require manual approval if quality score below 80",
            "Generate warnings for compliance violations",
            "Track and report technical debt trends"
        ]
        
        return {
            "quality_gates_configured": True,
            "gates_defined": len(quality_gates),
            "enforcement_rules": len(enforcement_rules),
            "automation_enabled": True,
            "notification_configured": True,
            "gates_configuration": quality_gates,
            "enforcement_details": enforcement_rules
        }
    
    async def _execute_monitoring_implementation(self) -> Dict:
        """Execute monitoring implementation"""
        
        # Setup continuous monitoring configuration
        monitoring_config = {
            "real_time_scanning": {
                "enabled": True,
                "scan_interval": "on_commit",
                "alert_thresholds": {
                    "new_high_vulnerabilities": 0,
                    "quality_score_drop": 5,
                    "compliance_violations": 3
                }
            },
            "dashboards": {
                "quality_overview": "configured",
                "security_status": "configured", 
                "compliance_tracking": "configured",
                "trend_analysis": "configured"
            },
            "integrations": {
                "ci_cd_pipeline": "enabled",
                "slack_notifications": "configured",
                "email_alerts": "configured",
                "jira_ticket_creation": "configured"
            }
        }
        
        # Setup alerting rules
        alert_rules = [
            "Critical: New high-severity vulnerability detected",
            "Warning: Quality score dropped below threshold",
            "Info: New compliance violations found",
            "Trend: Technical debt increasing over time"
        ]
        
        return {
            "monitoring_implemented": True,
            "real_time_scanning_enabled": True,
            "dashboards_created": len(monitoring_config["dashboards"]),
            "integrations_configured": len(monitoring_config["integrations"]),
            "alert_rules_defined": len(alert_rules),
            "monitoring_configuration": monitoring_config,
            "alert_rules": alert_rules
        }
    
    async def _compile_metrics(self, results: Dict) -> Dict:
        """Compile overall execution metrics"""
        phase_results = results["phase_results"]
        
        # Aggregate metrics from all phases
        overall_quality_score = 0
        total_vulnerabilities = 0
        compliance_score = 0
        technical_debt = 0
        
        if "static_analysis" in phase_results:
            static_results = phase_results["static_analysis"]
            overall_quality_score = static_results.get("overall_quality_score", 0)
            technical_debt = static_results.get("technical_debt_minutes", 0)
        
        if "security_scanning" in phase_results:
            security_results = phase_results["security_scanning"]
            total_vulnerabilities = security_results.get("total_vulnerabilities", 0)
        
        if "compliance_verification" in phase_results:
            compliance_results = phase_results["compliance_verification"]
            compliance_score = compliance_results.get("overall_compliance_score", 0)
        
        # Calculate overall health score
        health_score = (overall_quality_score + compliance_score) / 2
        if total_vulnerabilities > 0:
            health_score = health_score * 0.8  # Penalize for vulnerabilities
        
        return {
            "overall_quality_score": overall_quality_score,
            "security_vulnerabilities_found": total_vulnerabilities,
            "compliance_score": compliance_score,
            "technical_debt_minutes": technical_debt,
            "health_score": health_score,
            "quality_gates_configured": 4,
            "monitoring_enabled": True,
            "execution_time_minutes": 65,  # Simulated
            "issues_resolved": 0,  # Would track actual fixes
            "recommendations_provided": 15
        }

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    async def test_code_quality_agent():
        config = AgentConfiguration(
            agent_id="quality_agent_001",
            agent_type="code_quality",
            llm_provider=LLMProvider.OPENAI,
            llm_model="gpt-4",
            api_key="your-openai-api-key",
            enable_mcp=True,
            enable_a2a=True
        )
        
        agent = CodeQualityAgent(config)
        
        context = AgentContext(
            project_id="ecommerce_project_001",
            session_id="test_session_001", 
            workflow_id="test_workflow_001",
            shared_memory={
                "code_context": {
                    "repository_url": "https://github.com/company/ecommerce-app",
                    "primary_language": "python",
                    "frameworks": ["flask", "react"],
                    "quality_requirements": {
                        "minimum_quality_score": 80,
                        "security_compliance": "owasp_top_10",
                        "code_coverage": "85_percent"
                    }
                }
            }
        )
        
        task = {
            "type": "analyze_code_quality",
            "project_id": "ecommerce_project_001",
            "code_repository": {
                "url": "https://github.com/company/ecommerce-app",
                "branch": "main",
                "include_patterns": ["*.py", "*.js", "*.ts"]
            },
            "quality_standards": ["pep8", "owasp", "sonarqube"],
            "security_requirements": {
                "scan_dependencies": True,
                "check_secrets": True,
                "compliance_standards": ["owasp_top_10", "cwe_top_25"]
            },
            "quality_gates": {
                "minimum_quality_score": 80,
                "maximum_high_vulnerabilities": 0,
                "minimum_compliance": 85
            }
        }
        
        try:
            print("🛡️ Starting Code Quality Agent Test")
            print(f"Agent ID: {agent.agent_id}")
            print(f"Tools available: {[tool.name for tool in agent.tools]}")
            
            result = await agent.process(task, context)
            
            print("\n✅ Code Quality Agent Execution Complete!")
            print(f"Success: {result['success']}")
            print(f"Execution time: {result['execution_time']:.2f}s")
            
            if result['success']:
                reasoning = result['reasoning']
                print(f"\n🧠 Reasoning Summary:")
                print(f"  - Strategy: {reasoning['quality_strategy']['primary_focus']}")
                print(f"  - Automation: {reasoning['quality_strategy']['automation_level']}")
                print(f"  - Confidence: {reasoning['confidence_score']}")
                
                plan = result['plan']
                print(f"\n📋 Plan Summary:")
                print(f"  - Approach: {plan['approach']}")
                print(f"  - Phases: {len(plan['phases'])}")
                print(f"  - Total hours: {plan['estimated_total_hours']}")
                
                execution_result = result['result']
                if execution_result['success']:
                    metrics = execution_result['overall_metrics']
                    print(f"\n📊 Quality Analysis Results:")
                    print(f"  - Quality Score: {metrics['overall_quality_score']}")
                    print(f"  - Vulnerabilities: {metrics['security_vulnerabilities_found']}")
                    print(f"  - Compliance: {metrics['compliance_score']}%")
                    print(f"  - Health Score: {metrics['health_score']:.1f}")
                    
                    for phase_name, phase_result in execution_result['phase_results'].items():
                        print(f"\n  🔍 {phase_name.replace('_', ' ').title()}:")
                        if phase_name == "static_analysis":
                            print(f"    - Files analyzed: {phase_result.get('files_analyzed', 0)}")
                            print(f"    - Issues found: {phase_result.get('issues_found', 0)}")
                        elif phase_name == "security_scanning":
                            print(f"    - High risk: {phase_result.get('high_risk_vulnerabilities', 0)}")
                            print(f"    - Security score: {phase_result.get('security_score', 0)}")
                        elif phase_name == "compliance_verification":
                            print(f"    - Standards checked: {phase_result.get('standards_checked', 0)}")
                            print(f"    - Violations: {phase_result.get('total_violations', 0)}")
            
            else:
                print(f"❌ Execution failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(test_code_quality_agent())