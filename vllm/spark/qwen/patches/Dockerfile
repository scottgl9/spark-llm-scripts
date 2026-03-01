FROM avarok/dgx-vllm-nvfp4-kernel:v23
RUN apt-get update && apt-get install -y wget
RUN wget https://raw.githubusercontent.com/Avarok-Cybersecurity/dgx-vllm/main/fix_mtp_nvfp4_exclusion.py
RUN echo $(find / -name fix_mtp_nvfp4_exclusion.py)
RUN python3 $(find / -name fix_mtp_nvfp4_exclusion.py)
RUN echo "Ran fix_mtp_nvfp4_exclusion.py"
