#!/usr/bin/env python3
"""
Security Vulnerability Scanning Script

This script performs comprehensive security scanning including:
- Dependency vulnerability scanning using Safety
- License compliance checking
- Outdated package detection
- Security audit logging

Usage:
    python scripts/security_scan.py
    python scripts/security_scan.py --fix  # Attempt to fix vulnerabilities
    python scripts/security_scan.py --report  # Generate detailed report
"""

import os
import sys
import json
import subprocess
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config


class SecurityScanner:
    """Comprehensive security vulnerability scanner."""

    def __init__(self, log_file: str = "logs/security_scan.log"):
        """Initialize the security scanner."""
        self.log_file = log_file
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'vulnerabilities': [],
            'outdated_packages': [],
            'license_issues': [],
            'recommendations': []
        }

        # Set up logging
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging for security scans."""
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - SECURITY_SCAN - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Also log to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        self.logger = logging.getLogger(__name__)

    def scan_dependencies(self) -> Dict[str, Any]:
        """Scan dependencies for vulnerabilities using Safety."""
        self.logger.info("Scanning dependencies for vulnerabilities...")

        try:
            # Use requirements.txt file for faster scanning
            req_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'requirements.txt')

            if not os.path.exists(req_file):
                self.logger.warning("requirements.txt not found, skipping dependency scan")
                return {'status': 'skipped', 'error': 'requirements.txt not found'}

            # Run safety check on requirements.txt
            result = subprocess.run(
                [sys.executable, '-m', 'safety', 'check', '--file', req_file, '--json'],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )

            if result.returncode == 0:
                self.logger.info("✓ No known vulnerabilities found in dependencies")
                return {'status': 'clean', 'vulnerabilities': []}

            elif result.returncode in [64, 255]:  # Safety returns these codes for vulnerabilities
                try:
                    scan_result = json.loads(result.stdout)
                    vulnerabilities = scan_result.get('vulnerabilities', [])
                    self.results['vulnerabilities'] = vulnerabilities

                    if vulnerabilities:
                        self.logger.warning(f"⚠ Found {len(vulnerabilities)} vulnerabilities")
                        for vuln in vulnerabilities[:5]:  # Log first 5
                            vuln_id = vuln.get('vulnerability_id', 'Unknown')
                            pkg_name = vuln.get('package_name', 'Unknown')
                            self.logger.warning(f"  - {pkg_name}: Vulnerability {vuln_id}")

                        return {
                            'status': 'vulnerable',
                            'vulnerabilities': vulnerabilities,
                            'count': len(vulnerabilities)
                        }
                    else:
                        self.logger.info("✓ No known vulnerabilities found in dependencies")
                        return {'status': 'clean', 'vulnerabilities': []}

                except json.JSONDecodeError:
                    self.logger.error("Failed to parse safety output")
                    return {'status': 'error', 'error': 'Failed to parse safety output'}

            else:
                self.logger.error(f"Safety scan failed with return code {result.returncode}")
                self.logger.error(f"stdout: {result.stdout}")
                self.logger.error(f"stderr: {result.stderr}")
                return {'status': 'error', 'error': f'Return code {result.returncode}'}

        except FileNotFoundError:
            self.logger.error("Safety not installed. Install with: pip install safety")
            return {'status': 'error', 'error': 'Safety not installed'}

        except Exception as e:
            self.logger.error(f"Error during dependency scanning: {e}")
            return {'status': 'error', 'error': str(e)}

    def check_outdated_packages(self) -> Dict[str, Any]:
        """Check for outdated packages."""
        self.logger.info("Checking for outdated packages...")

        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json'],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )

            if result.returncode == 0:
                try:
                    outdated = json.loads(result.stdout)
                    self.results['outdated_packages'] = outdated

                    if outdated:
                        self.logger.warning(f"⚠ Found {len(outdated)} outdated packages")
                        for pkg in outdated[:5]:  # Log first 5
                            self.logger.warning(f"  - {pkg.get('name', 'Unknown')}: {pkg.get('version', 'Unknown')} -> {pkg.get('latest_version', 'Unknown')}")
                    else:
                        self.logger.info("✓ All packages are up to date")

                    return {
                        'status': 'success',
                        'outdated': outdated,
                        'count': len(outdated)
                    }

                except json.JSONDecodeError:
                    self.logger.error("Failed to parse pip output")
                    return {'status': 'error', 'error': 'Failed to parse pip output'}

            else:
                self.logger.error(f"Pip outdated check failed with return code {result.returncode}")
                return {'status': 'error', 'error': f'Return code {result.returncode}'}

        except Exception as e:
            self.logger.error(f"Error checking outdated packages: {e}")
            return {'status': 'error', 'error': str(e)}

    def check_licenses(self) -> Dict[str, Any]:
        """Check license compliance."""
        self.logger.info("Checking license compliance...")

        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--format=json'],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                packages = json.loads(result.stdout)

                # Check for packages with concerning licenses
                concerning_licenses = [
                    'GPL', 'LGPL', 'AGPL',  # Copyleft licenses that might be restrictive
                    'Proprietary', 'Commercial'  # Commercial licenses
                ]

                license_issues = []
                for pkg in packages:
                    license_info = pkg.get('license', '').upper()
                    for concerning in concerning_licenses:
                        if concerning in license_info:
                            license_issues.append({
                                'package': pkg.get('name', 'Unknown'),
                                'version': pkg.get('version', 'Unknown'),
                                'license': pkg.get('license', 'Unknown'),
                                'concern': f'Contains {concerning} license'
                            })
                            break

                self.results['license_issues'] = license_issues

                if license_issues:
                    self.logger.warning(f"⚠ Found {len(license_issues)} potential license issues")
                    for issue in license_issues[:3]:
                        self.logger.warning(f"  - {issue['package']}: {issue['license']}")
                else:
                    self.logger.info("✓ No concerning license issues found")

                return {
                    'status': 'success',
                    'license_issues': license_issues,
                    'count': len(license_issues)
                }

            else:
                self.logger.error(f"License check failed with return code {result.returncode}")
                return {'status': 'error', 'error': f'Return code {result.returncode}'}

        except Exception as e:
            self.logger.error(f"Error checking licenses: {e}")
            return {'status': 'error', 'error': str(e)}

    def generate_recommendations(self):
        """Generate security recommendations based on scan results."""
        recommendations = []

        # Vulnerability recommendations
        if self.results['vulnerabilities']:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Vulnerabilities',
                'action': 'Update vulnerable packages immediately',
                'details': f"Found {len(self.results['vulnerabilities'])} known vulnerabilities"
            })

        # Outdated package recommendations
        if self.results['outdated_packages']:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Updates',
                'action': 'Update outdated packages regularly',
                'details': f"{len(self.results['outdated_packages'])} packages have available updates"
            })

        # License recommendations
        if self.results['license_issues']:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Licensing',
                'action': 'Review license compatibility',
                'details': f"{len(self.results['license_issues'])} packages have potentially restrictive licenses"
            })

        # General recommendations
        recommendations.extend([
            {
                'priority': 'LOW',
                'category': 'Monitoring',
                'action': 'Run security scans weekly',
                'details': 'Regular scanning helps catch new vulnerabilities'
            },
            {
                'priority': 'LOW',
                'category': 'Dependencies',
                'action': 'Audit third-party dependencies',
                'details': 'Regularly review and minimize external dependencies'
            }
        ])

        self.results['recommendations'] = recommendations
        return recommendations

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive security report."""
        report = []
        report.append("=" * 60)
        report.append("SECURITY SCAN REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {self.results['timestamp']}")
        report.append("")

        # Vulnerabilities
        report.append("DEPENDENCY VULNERABILITIES")
        report.append("-" * 30)
        if self.results['vulnerabilities']:
            for vuln in self.results['vulnerabilities']:
                report.append(f"Package: {vuln.get('package', 'Unknown')}")
                report.append(f"Vulnerability: {vuln.get('vulnerability', 'Unknown')}")
                report.append(f"Severity: {vuln.get('severity', 'Unknown')}")
                report.append("")
        else:
            report.append("✓ No known vulnerabilities found")
        report.append("")

        # Outdated packages
        report.append("OUTDATED PACKAGES")
        report.append("-" * 20)
        if self.results['outdated_packages']:
            for pkg in self.results['outdated_packages'][:10]:  # Limit to 10
                report.append(f"{pkg.get('name', 'Unknown')}: {pkg.get('version', 'Unknown')} -> {pkg.get('latest_version', 'Unknown')}")
        else:
            report.append("✓ All packages are up to date")
        report.append("")

        # License issues
        report.append("LICENSE ISSUES")
        report.append("-" * 15)
        if self.results['license_issues']:
            for issue in self.results['license_issues']:
                report.append(f"{issue['package']}: {issue['license']} ({issue['concern']})")
        else:
            report.append("✓ No concerning license issues found")
        report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 15)
        for rec in self.results['recommendations']:
            report.append(f"[{rec['priority']}] {rec['category']}: {rec['action']}")
            report.append(f"  {rec['details']}")
            report.append("")

        report.append("=" * 60)

        report_text = "\n".join(report)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Report saved to {output_file}")

        return report_text

    def run_full_scan(self) -> Dict[str, Any]:
        """Run a complete security scan."""
        self.logger.info("Starting comprehensive security scan...")

        # Run all scans
        vuln_result = self.scan_dependencies()
        outdated_result = self.check_outdated_packages()
        license_result = self.check_licenses()

        # Generate recommendations
        self.generate_recommendations()

        # Determine overall status
        if vuln_result.get('status') == 'vulnerable':
            overall_status = 'VULNERABLE'
        elif vuln_result.get('status') == 'error' or outdated_result.get('status') == 'error' or license_result.get('status') == 'error':
            overall_status = 'ERROR'
        else:
            overall_status = 'CLEAN'

        self.results['overall_status'] = overall_status

        self.logger.info(f"Security scan completed with status: {overall_status}")

        return {
            'status': overall_status,
            'vulnerabilities': vuln_result,
            'outdated': outdated_result,
            'licenses': license_result,
            'recommendations': self.results['recommendations']
        }


def main():
    """Main entry point for security scanning."""
    import argparse

    parser = argparse.ArgumentParser(description='Security Vulnerability Scanner')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix vulnerabilities')
    parser.add_argument('--report', type=str, help='Generate detailed report to file')
    parser.add_argument('--quiet', action='store_true', help='Suppress console output')

    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        logging.getLogger('').handlers[1].setLevel(logging.ERROR)  # Suppress console output

    # Run security scan
    scanner = SecurityScanner()

    try:
        results = scanner.run_full_scan()

        # Generate report if requested
        if args.report:
            scanner.generate_report(args.report)

        # Attempt fixes if requested
        if args.fix and results['vulnerabilities'].get('status') == 'vulnerable':
            print("\nAttempting to fix vulnerabilities...")
            print("Note: This will attempt to update packages, which may break compatibility")
            print("Consider testing in a separate environment first")

            # This would require more sophisticated fix logic
            print("Manual intervention required for vulnerability fixes")

        # Exit with appropriate code
        if results['status'] == 'VULNERABLE':
            sys.exit(1)  # Vulnerabilities found
        elif results['status'] == 'ERROR':
            sys.exit(2)  # Scan failed
        else:
            sys.exit(0)  # Clean

    except KeyboardInterrupt:
        print("\nScan interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Scan failed with error: {e}")
        sys.exit(3)


if __name__ == '__main__':
    main()