#!/usr/bin/env python3
"""
FinRL Tutorial Environment Setup Script
환경 설정 자동화 스크립트
"""

import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path

class EnvironmentSetup:
    def __init__(self):
        self.log_file = Path("setup_log.json")
        self.logs = []

    def log_step(self, step, status, message=""):
        """단계별 로그 기록"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "status": status,
            "message": message
        }
        self.logs.append(entry)
        print(f"[{status}] {step}: {message}")

    def save_log(self):
        """로그를 JSON 파일로 저장"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)

    def check_python_version(self):
        """Python 버전 확인"""
        version = sys.version_info
        self.log_step(
            "Python Version Check",
            "INFO",
            f"Python {version.major}.{version.minor}.{version.micro}"
        )

        if version.major != 3 or version.minor < 8:
            self.log_step(
                "Python Version Check",
                "ERROR",
                "Python 3.8 이상이 필요합니다"
            )
            return False
        return True

    def install_packages(self):
        """패키지 설치"""
        self.log_step("Package Installation", "START", "패키지 설치 시작")

        try:
            # requirements.txt 파일 확인
            req_file = Path("requirements.txt")
            if not req_file.exists():
                self.log_step(
                    "Package Installation",
                    "ERROR",
                    "requirements.txt 파일을 찾을 수 없습니다"
                )
                return False

            # pip 업그레이드
            self.log_step("Package Installation", "INFO", "pip 업그레이드 중...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ])

            # requirements.txt 설치
            self.log_step("Package Installation", "INFO", "필수 패키지 설치 중...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])

            self.log_step("Package Installation", "SUCCESS", "모든 패키지 설치 완료")
            return True

        except subprocess.CalledProcessError as e:
            self.log_step(
                "Package Installation",
                "ERROR",
                f"설치 중 오류 발생: {str(e)}"
            )
            return False

    def verify_installation(self):
        """설치 검증"""
        self.log_step("Installation Verification", "START", "설치 검증 시작")

        required_packages = [
            "finrl",
            "yfinance",
            "stable_baselines3",
            "gymnasium",
            "pandas",
            "numpy",
            "matplotlib"
        ]

        failed_packages = []

        for package in required_packages:
            try:
                __import__(package)
                self.log_step(
                    "Installation Verification",
                    "SUCCESS",
                    f"{package} 정상 작동"
                )
            except ImportError as e:
                failed_packages.append(package)
                self.log_step(
                    "Installation Verification",
                    "ERROR",
                    f"{package} import 실패: {str(e)}"
                )

        if failed_packages:
            self.log_step(
                "Installation Verification",
                "ERROR",
                f"실패한 패키지: {', '.join(failed_packages)}"
            )
            return False

        self.log_step("Installation Verification", "SUCCESS", "모든 패키지 검증 완료")
        return True

    def run(self):
        """전체 설정 프로세스 실행"""
        print("="*60)
        print("FinRL Tutorial 환경 설정 시작")
        print("="*60)

        if not self.check_python_version():
            self.save_log()
            return False

        if not self.install_packages():
            self.save_log()
            return False

        if not self.verify_installation():
            self.save_log()
            return False

        self.log_step("Setup Complete", "SUCCESS", "환경 설정이 완료되었습니다!")
        self.save_log()

        print("="*60)
        print("설정 완료! 로그는 setup_log.json 파일에 저장되었습니다.")
        print("="*60)
        return True

if __name__ == "__main__":
    setup = EnvironmentSetup()
    success = setup.run()
    sys.exit(0 if success else 1)
