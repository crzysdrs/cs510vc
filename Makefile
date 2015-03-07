all : stitcher

stitcher : stitcher.cpp
	g++ $< `pkg-config --libs opencv` -o $@

result.jpg : stitcher
	git annex get annex/stitch/*
	./stitcher annex/stitch/*
