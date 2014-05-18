#version 420 core

// The vertex shader receives the current particle (A) and its successor (B).
struct PARTICLE
{
	layout(location = 0) vec2 PositionA;
	layout(location = 1) uint StateA;	// 2=Head, 1=Body, 0=Tail

	layout(location = 2) vec2 PositionB;
	layout(location = 3) uint StateB;	// 2=Head, 1=Body, 0=Tail
};

// input from input assembler.
in PARTICLE vs_in;

// output of the vertex shader.
out PARTICLE vs_out;

void main()
{
	vs_out = vs_in;
}
