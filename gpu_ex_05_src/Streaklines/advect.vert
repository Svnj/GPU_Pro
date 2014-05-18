#version 420 core

// The vertex shader receives the current particle (A) and its successor (B).
//struct PARTICLE
//{
        layout(location = 0) in vec2 PositionA;
        layout(location = 1) in uint StateA;	// 2=Head, 1=Body, 0=Tail

        layout(location = 2) in vec2 PositionB;
        layout(location = 3) in uint StateB;	// 2=Head, 1=Body, 0=Tail
//};

struct PARTICLE
{
        vec2 PositionA;
        uint StateA;	// 2=Head, 1=Body, 0=Tail

        vec2 PositionB;
        uint StateB;	// 2=Head, 1=Body, 0=Tail
};

// input from input assembler.
//in PARTICLE vs_in;

// output of the vertex shader.
out PARTICLE vs_out;

void main()
{
        vs_out.PositionA = PositionA;
        vs_out.StateA = StateA;
        vs_out.PositionB = PositionB;
        vs_out.StateB = StateB;
}
