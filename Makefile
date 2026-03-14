reinstall_package:
	@pip uninstall -y renal_sight || :
	@pip install -e .

run_api:
	uvicorn renal_sight.api.fast:app --reload
