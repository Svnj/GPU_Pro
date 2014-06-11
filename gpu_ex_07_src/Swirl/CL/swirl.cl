#define DIM 512
#define blockSize 8

#define PI 3.1415926535897932f
#define centerX (DIM/2)
#define centerY (DIM/2)

float alpha(float r,float a,float b)
{
    return a * pow(r,b);
}

kernel void swirlKernel(global float *sourcePtr,global float *targetPtr, float a,float b)
{
    // DONE: Index berechnen
    int index = (get_global_size(0) * get_global_id(1)) + get_global_id(0);

    // TODO: Den swirl invertieren.
    float2 center = (float2)(centerX,centerY);
    float2 upVector = (float2)(0,1);
    float2 startVector = (float2)(get_global_id(0),get_global_id(1)) - center;
    float vectorLength = length(startVector);
    upVector *= vectorLength;
    float angleBetweenUpVectorAndStartVector = (dot(startVector,upVector)/(vectorLength*vectorLength));
    angleBetweenUpVectorAndStartVector = acos(angleBetweenUpVectorAndStartVector);
    angleBetweenUpVectorAndStartVector = get_global_id(0) > 255 ? (2*PI) - angleBetweenUpVectorAndStartVector : angleBetweenUpVectorAndStartVector;
    angleBetweenUpVectorAndStartVector = degrees(angleBetweenUpVectorAndStartVector);
    float turnBackAngle = alpha(vectorLength,a,b);
    //turnBackAngle = fmod(turnBackAngle,360.0f);
    float2 resultPosition = (float2)(
                                startVector.x * cos(radians(turnBackAngle)) - startVector.y * sin(radians(turnBackAngle)),
                                startVector.x * sin(radians(turnBackAngle)) + startVector.y * cos(radians(turnBackAngle))
                            );
    //resultPosition = fabs(resultPosition);

    resultPosition += center;
    int getterIndex = (DIM * resultPosition.y) + resultPosition.x;

    if(getterIndex >= 0 && getterIndex <= DIM*DIM)
        targetPtr[index] = sourcePtr[getterIndex];
    else
        targetPtr[index] = 0.0f;
}

//    //float2 zeroVec = (float2)(0.0f,0.0f);
//    //float maxDistance = distance(center,zeroVec);

//    float2 position = (float2)(get_global_id(0),get_global_id(1));
//    float2 A = position - center;
//    float radius = distance(position,center);

//    float alpha = alpha(radius,a,b);

//    float2 b = (cos(alpha) * radius * radius)/

//    radius /= maxDistance;

//    targetPtr[index] = sourcePtr[(int)(test)];//radius;    // simple copy

double calculateOriginalAngle(double2 rel)
{
    double originalAngle;
    // Calculate the angle our points are relative to UV origin. Everything is in radians.
    if (rel.x != 0)
    {
            originalAngle = atan(fabs(rel.y)/fabs(rel.x));

            if ( rel.x > 0 && rel.y < 0) originalAngle = 2.0f*PI - originalAngle;

            else if (rel.x <= 0 && rel.y >=0) originalAngle = PI-originalAngle;

            else if (rel.x <=0 && rel.y <0) originalAngle += PI;
    }

    else
    {
            // Take care of rare special case
            if (rel.y >= 0) originalAngle = 0.5f * PI;

            else originalAngle = 1.5f * PI;
    }

    return originalAngle;
}

int2 myClamp(int2 src,int width,int height)
{
    src.x = clamp(src.x,0,width);
    src.y = clamp(src.y,0,height);
    return src;
}

kernel void swirlKernelSCB(global float *sourcePtr,global float *targetPtr, float a,float b)
{
    // DONE: Index berechnen
    int index = (get_global_size(0) * get_global_id(1)) + get_global_id(0);

    double2 rel = (double2)((double)get_global_id(0)-centerX,((double)centerY-get_global_id(1)));
    // rel.x and rel.y are points in our UV space

    double originalAngle = calculateOriginalAngle(rel);

    // Calculate the distance from the center of the UV using pythagorean distance
    double radius = length(rel);

    // Use any equation we want to determine how much to rotate image by
    double newAngle = originalAngle + alpha(radius,a,b);

    // Transform source UV coordinates back into bitmap coordinates
    int2 src = (int2)((floor(radius * cos(newAngle)+0.5f)),(floor(radius * sin(newAngle)+0.5f)));

    src.x += centerX;
    src.y += centerY;
    src.y = DIM - src.y;

    // Clamp the source to legal image pixel
    src = myClamp(src,DIM,DIM);

    int getterIndex = (DIM * src.y) + src.x;

    targetPtr[index] = sourcePtr[getterIndex];
}
