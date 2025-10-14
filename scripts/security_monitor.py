#!/usr/bin/env python3
"""
Security Monitoring and Alerting System

This script monitors security logs and metrics for anomalous patterns
and sends alerts when security incidents are detected.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config


class SecurityMonitor:
    """Monitors security events and detects anomalous patterns."""

    def __init__(self, config=None):
        """Initialize the security monitor."""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)

        # Monitoring data structures
        self.upload_patterns = defaultdict(lambda: deque(maxlen=1000))  # IP -> upload times
        self.validation_failures = defaultdict(int)  # IP -> failure count
        self.rate_limit_hits = defaultdict(int)  # IP -> rate limit hits
        self.file_size_history = deque(maxlen=1000)  # Recent file sizes
        self.error_patterns = defaultdict(int)  # Error type -> count

        # Alert thresholds
        self.thresholds = {
            'uploads_per_minute': 10,  # Max uploads per minute per IP
            'validation_failures_per_hour': 5,  # Max validation failures per hour per IP
            'rate_limit_hits_per_hour': 3,  # Max rate limit hits per hour per IP
            'large_file_percentage': 0.1,  # Percentage of uploads that are large files
            'error_rate_threshold': 0.05  # 5% error rate threshold
        }

        # Alert state
        self.alerts_sent = set()  # Track sent alerts to avoid spam
        self.last_alert_time = {}  # Alert type -> last sent time

    def analyze_upload_patterns(self, log_entry: Dict[str, Any]) -> List[str]:
        """Analyze upload patterns for anomalies."""
        alerts = []

        if 'client_ip' in log_entry:
            client_ip = log_entry['client_ip']
            current_time = datetime.now()

            # Track upload frequency
            self.upload_patterns[client_ip].append(current_time)

            # Check for rapid uploads (potential DoS)
            recent_uploads = [t for t in self.upload_patterns[client_ip]
                            if current_time - t < timedelta(minutes=1)]

            if len(recent_uploads) > self.thresholds['uploads_per_minute']:
                alert_key = f"rapid_uploads_{client_ip}"
                if self._should_send_alert(alert_key):
                    alerts.append(f"High upload frequency from {client_ip}: {len(recent_uploads)} uploads/minute")
                    self._mark_alert_sent(alert_key)

        # Check file size patterns
        if 'file_size' in log_entry:
            file_size = log_entry['file_size']
            self.file_size_history.append(file_size)

            # Check for unusually large files
            if len(self.file_size_history) > 10:
                avg_size = sum(self.file_size_history) / len(self.file_size_history)
                large_files = [s for s in self.file_size_history if s > avg_size * 2]

                if len(large_files) / len(self.file_size_history) > self.thresholds['large_file_percentage']:
                    alert_key = "large_file_spike"
                    if self._should_send_alert(alert_key):
                        alerts.append(f"Unusual spike in large file uploads: {len(large_files)}/{len(self.file_size_history)} files")
                        self._mark_alert_sent(alert_key)

        return alerts

    def analyze_validation_failures(self, log_entry: Dict[str, Any]) -> List[str]:
        """Analyze validation failures for attack patterns."""
        alerts = []
        client_ip = log_entry.get('client_ip', 'unknown')

        if client_ip != 'unknown':
            self.validation_failures[client_ip] += 1

            # Check for high validation failure rate
            if self.validation_failures[client_ip] > self.thresholds['validation_failures_per_hour']:
                alert_key = f"validation_failures_{client_ip}"
                if self._should_send_alert(alert_key):
                    alerts.append(f"High validation failure rate from {client_ip}: {self.validation_failures[client_ip]} failures")
                    self._mark_alert_sent(alert_key)

        # Check for specific attack patterns
        if 'validation_type' in log_entry:
            validation_type = log_entry['validation_type']
            input_value = log_entry.get('input_value', '')

            # Check for path traversal attempts
            if validation_type == 'filename' and ('../' in input_value or '..' in input_value):
                alert_key = f"path_traversal_{client_ip}"
                if self._should_send_alert(alert_key):
                    alerts.append(f"Path traversal attempt detected from {client_ip}: {input_value[:100]}")
                    self._mark_alert_sent(alert_key)

            # Check for SQL injection attempts
            if 'input_value' in log_entry and any(sql_pattern in input_value.upper() for sql_pattern in
                ['UNION SELECT', 'DROP TABLE', 'INSERT INTO', 'UPDATE ', 'DELETE FROM']):
                alert_key = f"sql_injection_{client_ip}"
                if self._should_send_alert(alert_key):
                    alerts.append(f"Potential SQL injection attempt from {client_ip}: {input_value[:100]}")
                    self._mark_alert_sent(alert_key)

        return alerts

    def analyze_rate_limits(self, log_entry: Dict[str, Any]) -> List[str]:
        """Analyze rate limiting patterns."""
        alerts = []

        if 'client_ip' in log_entry:
            client_ip = log_entry['client_ip']
            self.rate_limit_hits[client_ip] += 1

            if self.rate_limit_hits[client_ip] > self.thresholds['rate_limit_hits_per_hour']:
                alert_key = f"rate_limit_spike_{client_ip}"
                if self._should_send_alert(alert_key):
                    alerts.append(f"Persistent rate limit violations from {client_ip}: {self.rate_limit_hits[client_ip]} hits")
                    self._mark_alert_sent(alert_key)

        return alerts

    def analyze_error_patterns(self, recent_logs: List[Dict[str, Any]]) -> List[str]:
        """Analyze error patterns across all logs."""
        alerts = []

        total_requests = len(recent_logs)
        if total_requests == 0:
            return alerts

        error_count = sum(1 for log in recent_logs if 'error' in log or log.get('event_type', '').endswith('_FAILED'))

        error_rate = error_count / total_requests

        if error_rate > self.thresholds['error_rate_threshold']:
            alert_key = "high_error_rate"
            if self._should_send_alert(alert_key):
                alerts.append(f"High error rate detected: {error_rate:.1%} ({error_count}/{total_requests} requests)")
                self._mark_alert_sent(alert_key)

        return alerts

    def _should_send_alert(self, alert_key: str) -> bool:
        """Check if an alert should be sent (to prevent spam)."""
        current_time = datetime.now()

        # Don't send the same alert more than once every 5 minutes
        if alert_key in self.last_alert_time:
            time_since_last = current_time - self.last_alert_time[alert_key]
            if time_since_last < timedelta(minutes=5):
                return False

        return True

    def _mark_alert_sent(self, alert_key: str):
        """Mark an alert as sent."""
        self.last_alert_time[alert_key] = datetime.now()

    def process_log_entry(self, log_entry: Dict[str, Any]) -> List[str]:
        """Process a single log entry and return any alerts."""
        alerts = []

        # Extract common fields
        if 'client_ip' not in log_entry and 'ip' in log_entry:
            log_entry['client_ip'] = log_entry['ip']

        # Analyze based on log type
        event_type = log_entry.get('event_type', '')

        if 'UPLOAD' in event_type:
            alerts.extend(self.analyze_upload_patterns(log_entry))
        elif 'VALIDATION_FAILURE' in event_type:
            alerts.extend(self.analyze_validation_failures(log_entry))
        elif 'RATE_LIMIT' in event_type:
            alerts.extend(self.analyze_rate_limits(log_entry))

        return alerts

    def monitor_logs(self, log_file: str, continuous: bool = False) -> List[str]:
        """Monitor log file for security events."""
        alerts = []

        try:
            with open(log_file, 'r') as f:
                if continuous:
                    # Monitor continuously (for production use)
                    f.seek(0, 2)  # Go to end of file
                    while True:
                        line = f.readline()
                        if line:
                            try:
                                log_entry = json.loads(line.strip())
                                new_alerts = self.process_log_entry(log_entry)
                                alerts.extend(new_alerts)
                            except json.JSONDecodeError:
                                continue
                        else:
                            time.sleep(1)  # Wait for new lines
                else:
                    # Process existing log file
                    recent_logs = []
                    for line in f:
                        try:
                            log_entry = json.loads(line.strip())
                            new_alerts = self.process_log_entry(log_entry)
                            alerts.extend(new_alerts)
                            recent_logs.append(log_entry)
                        except json.JSONDecodeError:
                            continue

                    # Analyze error patterns
                    if len(recent_logs) > 10:
                        alerts.extend(self.analyze_error_patterns(recent_logs[-100:]))  # Last 100 entries

        except FileNotFoundError:
            self.logger.warning(f"Log file not found: {log_file}")
        except Exception as e:
            self.logger.error(f"Error monitoring logs: {e}")

        return alerts


class AlertManager:
    """Manages sending security alerts."""

    def __init__(self, config=None):
        """Initialize the alert manager."""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)

    def send_email_alert(self, subject: str, message: str, recipients: List[str]):
        """Send email alert."""
        try:
            # Email configuration (would be in config)
            smtp_server = os.getenv('SMTP_SERVER', 'localhost')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            smtp_user = os.getenv('SMTP_USER')
            smtp_pass = os.getenv('SMTP_PASS')

            if not smtp_user:
                self.logger.warning("SMTP not configured, skipping email alert")
                return

            msg = MIMEMultipart()
            msg['From'] = smtp_user
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"SECURITY ALERT: {subject}"

            msg.attach(MIMEText(message, 'plain'))

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            if smtp_pass:
                server.login(smtp_user, smtp_pass)
            text = msg.as_string()
            server.sendmail(smtp_user, recipients, text)
            server.quit()

            self.logger.info(f"Security alert email sent to {recipients}")

        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")

    def send_slack_alert(self, message: str, webhook_url: Optional[str] = None):
        """Send Slack alert."""
        try:
            final_webhook_url = webhook_url or os.getenv('SLACK_WEBHOOK_URL')
            if not final_webhook_url:
                self.logger.warning("Slack webhook not configured, skipping Slack alert")
                return

            import requests
            payload = {
                "text": f"🚨 SECURITY ALERT 🚨\n{message}",
                "username": "Security Monitor",
                "icon_emoji": ":shield:"
            }

            response = requests.post(final_webhook_url, json=payload)
            if response.status_code == 200:
                self.logger.info("Security alert sent to Slack")
            else:
                self.logger.error(f"Failed to send Slack alert: {response.status_code}")

        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")

    def send_alerts(self, alerts: List[str]):
        """Send alerts through configured channels."""
        if not alerts:
            return

        alert_message = "\n".join(f"• {alert}" for alert in alerts)
        subject = f"Security Alert - {len(alerts)} incident(s) detected"

        # Send email alert
        email_recipients = os.getenv('SECURITY_ALERT_EMAILS', '').split(',')
        if email_recipients and email_recipients[0]:
            self.send_email_alert(subject, alert_message, email_recipients)

        # Send Slack alert
        slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        if slack_webhook:
            self.send_slack_alert(alert_message, slack_webhook)

        # Log alerts locally
        for alert in alerts:
            self.logger.warning(f"SECURITY ALERT: {alert}")


def main():
    """Main monitoring function."""
    import argparse

    parser = argparse.ArgumentParser(description='Security Monitoring and Alerting')
    parser.add_argument('--log-file', default='logs/security_events.log', help='Security log file to monitor')
    parser.add_argument('--continuous', action='store_true', help='Monitor continuously')
    parser.add_argument('--no-alerts', action='store_true', help='Disable sending alerts')

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Initialize monitoring
    monitor = SecurityMonitor()
    alert_manager = AlertManager()

    try:
        logger.info(f"Starting security monitoring of {args.log_file}")

        while True:
            # Monitor logs
            alerts = monitor.monitor_logs(args.log_file, continuous=args.continuous)

            # Send alerts if any
            if alerts and not args.no_alerts:
                alert_manager.send_alerts(alerts)

            if not args.continuous:
                break

            # Wait before next check
            time.sleep(60)  # Check every minute

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Monitoring failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()