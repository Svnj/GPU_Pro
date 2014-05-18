
/* ==============================================================
	Amira Loader by Tino Weinkauf.
	http://www.mpi-inf.mpg.de/~weinkauf/notes/amiramesh.html
   ==============================================================  */

#include <stdio.h>
#include <string.h>
#include <assert.h>

/** Find a string in the given buffer and return a pointer
    to the contents directly behind the SearchString.
    If not found, return the buffer. A subsequent sscanf()
    will fail then, but at least we return a decent pointer.
*/
const char* FindAndJump(const char* buffer, const char* SearchString)
{
    const char* FoundLoc = strstr(buffer, SearchString);
    if (FoundLoc) return FoundLoc + strlen(SearchString);
    return buffer;
}


/** A simple routine to read an AmiraMesh file
    that defines a scalar/vector field on a uniform grid.
*/
float* LoadField(const char* FileName, 
	int* xDim, int* yDim, int* zDim, 
	float* xmin, float* ymin, float* zmin, 
	float* xmax, float* ymax, float* zmax)
{
	float* pData = NULL;
    FILE* fp = fopen(FileName, "rb");
    if (!fp)
    {
        printf("Could not find %s\n", FileName);
        return NULL;
    }

    printf("Reading %s\n", FileName);

    //We read the first 2k bytes into memory to parse the header.
    //The fixed buffer size looks a bit like a hack, and it is one, but it gets the job done.
    char buffer[2048];
    fread(buffer, sizeof(char), 2047, fp);
    buffer[2047] = '\0'; //The following string routines prefer null-terminated strings

    if (!strstr(buffer, "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1"))
    {
        printf("Not a proper AmiraMesh file.\n");
        fclose(fp);
        return NULL;
    }

    //Find the Lattice definition, i.e., the dimensions of the uniform grid    
    sscanf(FindAndJump(buffer, "define Lattice"), "%d %d %d", xDim, yDim, zDim);
    printf("\tGrid Dimensions: %d %d %d\n", *xDim, *yDim, *zDim);

    //Find the BoundingBox    
    sscanf(FindAndJump(buffer, "BoundingBox"), "%g %g %g %g %g %g", xmin, xmax, ymin, ymax, zmin, zmax);
    printf("\tBoundingBox in x-Direction: [%g ... %g]\n", *xmin, *xmax);
    printf("\tBoundingBox in y-Direction: [%g ... %g]\n", *ymin, *ymax);
    printf("\tBoundingBox in z-Direction: [%g ... %g]\n", *zmin, *zmax);

    //Is it a uniform grid? We need this only for the sanity check below.
    const bool bIsUniform = (strstr(buffer, "CoordType \"uniform\"") != NULL);
    printf("\tGridType: %s\n", bIsUniform ? "uniform" : "UNKNOWN");

    //Type of the field: scalar, vector
    int NumComponents(0);
    if (strstr(buffer, "Lattice { float Data }"))
    {
        //Scalar field
        NumComponents = 1;
    }
    else
    {
        //A field with more than one component, i.e., a vector field
        sscanf(FindAndJump(buffer, "Lattice { float["), "%d", &NumComponents);
    }
    printf("\tNumber of Components: %d\n", NumComponents);

    //Sanity check
    if (*xDim <= 0 || *yDim <= 0 || *zDim <= 0
        || *xmin > *xmax || *ymin > *ymax || *zmin > *zmax
        || !bIsUniform || NumComponents <= 0)
    {
        printf("Something went wrong\n");
        fclose(fp);
        return NULL;
    }

    //Find the beginning of the data section
    const long idxStartData = strstr(buffer, "# Data section follows") - buffer;
    if (idxStartData > 0)
    {
        //Set the file pointer to the beginning of "# Data section follows"
        fseek(fp, idxStartData, SEEK_SET);
        //Consume this line, which is "# Data section follows"
        fgets(buffer, 2047, fp);
        //Consume the next line, which is "@1"
        fgets(buffer, 2047, fp);

        //Read the data
        // - how much to read
        const size_t NumToRead = *xDim * *yDim * *zDim * NumComponents;
        // - prepare memory; use malloc() if you're using pure C
        pData = new float[NumToRead];
        if (pData)
        {
            // - do it
            const size_t ActRead = fread((void*)pData, sizeof(float), NumToRead, fp);
            // - ok?
            if (NumToRead != ActRead)
            {
                printf("Something went wrong while reading the binary data section.\nPremature end of file?\n");
                delete[] pData;
                fclose(fp);
                return NULL;
            }

			/*
            //Test: Print all data values
            //Note: Data runs x-fastest, i.e., the loop over the x-axis is the innermost
            printf("\nPrinting all values in the same order in which they are in memory:\n");
            int Idx(0);
            for(int k=0;k<zDim;k++)
            {
                for(int j=0;j<yDim;j++)
                {
                    for(int i=0;i<xDim;i++)
                    {
                        //Note: Random access to the value (of the first component) of the grid point (i,j,k):
                        // pData[((k * yDim + j) * xDim + i) * NumComponents]
                        assert(pData[((k * yDim + j) * xDim + i) * NumComponents] == pData[Idx * NumComponents]);

                        for(int c=0;c<NumComponents;c++)
                        {
                            printf("%g ", pData[Idx * NumComponents + c]);
                        }
                        printf("\n");
                        Idx++;
                    }
                }
            }

            delete[] pData; */
        }
    }

    fclose(fp);
	return pData;
}