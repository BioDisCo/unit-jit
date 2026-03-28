.DEFAULT_GOAL := version

version:
	@uv version

release:
	@test -n "$(v)" || (echo "current version: $$(uv version)" && echo "usage: make release v=X.Y.Z" && exit 1)
	uv version $(v)
	git add pyproject.toml
	git commit -m "release $(v)"
	git tag v$(v)
	git push origin main v$(v)
