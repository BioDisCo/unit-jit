.DEFAULT_GOAL := version

version:
	@uv version

release:
	@test -n "$(v)" || (echo "usage: make release v=0.4.4" && exit 1)
	uv version $(v)
	git add pyproject.toml
	git commit -m "release $(v)"
	git tag v$(v)
	git push origin main v$(v)
