# Image Retrieval System

## 1. Introduction
This project is a comprehensive Content-Based Image Retrieval (CBIR) system developed in Java. It is designed to find similar images within a dataset based on their visual content rather than metadata or tags. The system employs a sophisticated, multi-tiered feature extraction strategy, leveraging a variety of computer vision and deep learning libraries to generate descriptive feature vectors for each image. These vectors are then indexed using efficient data structures to enable fast and accurate similarity searches.

The core of the system lies in its flexible and extensible architecture. It utilizes a Strategy design pattern for both feature extraction and searching, allowing new algorithms to be seamlessly integrated. A rule engine dynamically selects the most appropriate feature extractor for a given image based on its properties, such as file size, resolution, and EXIF metadata. This intelligent selection process ensures a balance between computational efficiency and retrieval accuracy. For instance, lightweight classical algorithms like ORB are used for small, low-resolution images, while powerful deep learning models (ResNet50 via DJL and DL4J) are reserved for high-resolution images from professional cameras, where capturing subtle semantic features is crucial.

For searching, the system offers multiple strategies, including a brute-force linear scan for deep learning embeddings and a more optimized k-d tree-based search for approximate nearest neighbor queries. This dual-strategy approach allows the user to choose between exhaustive accuracy and high-speed retrieval, making the system adaptable to various use cases, from real-time applications to large-scale offline indexing. The project is built with modern Java practices, incorporating robust error handling, logging, and a clear separation of concerns to facilitate maintenance and future development.

## 2. Features

**Dynamic Feature Extraction Strategy**: A rule engine analyzes image characteristics (file size, resolution, camera metadata) to select the most suitable feature extraction algorithm.

**Multiple Extractor Implementations**:

- **ORB (Oriented FAST and Rotated BRIEF)**: A fast and efficient classical feature detector and descriptor, ideal for quick processing of lower-quality images. Implemented using OpenCV.
- **BoofCV SURF**: A robust classical feature extractor, serving as a reliable default.
- **DJL (Deep Java Library) ResNet50**: Leverages a pre-trained deep learning model for high-level semantic feature extraction.
- **DL4J (Deeplearning4j) ResNet50**: An alternative deep learning implementation, chosen specifically for images with professional camera metadata.

**Flexible Search Mechanisms**:

- **DeepMetricSearch**: Performs an exhaustive k-nearest neighbor search, perfect for the high-dimensional vectors produced by deep learning models. It is thread-safe and supports dynamic insertion of new features.
- **BestBinFirstSearch**: Implements an approximate nearest neighbor search using a K-D Tree, offering a significant speed advantage for large datasets where perfect accuracy is not strictly required.

**Modular and Extensible Architecture**: The use of interfaces (`Extractable`, `Searchable`, `Buildable`, `Insertable`) and the Strategy pattern makes it easy to add new extraction or search algorithms without modifying existing code.

**Advanced Utility Functions**: A comprehensive `FeatureUtils` class provides essential vector operations, including L2 normalization and multiple distance metric calculations (Cosine, Euclidean, Manhattan).

**Robust Indexing**:

- The `KDTreeBuilder` class constructs a K-D Tree from image features for efficient spatial partitioning and searching.
- The `DeepMetricSearch` class provides a simple in-memory list-based index that can be built and added to dynamically.

## 3. System Architecture

The system is organized into several key packages, each with a distinct responsibility:

### `main.retrieval.features`: The heart of the feature extraction logic.

- **ExtractorFactory**: A factory class that acts as the primary entry point for feature extraction. It uses the `RuleEngine` to delegate the choice of extractor.
- **extractor**: Contains implementations of the `Extractable` interface (e.g., `ORBExtractor`, `DJLExtractor`). Each class is responsible for a single feature extraction algorithm.
- **rules**: Houses the `RuleEngine` and the individual `ExtractorRule` implementations (`FileSizeRule`, `ResolutionRule`, etc.). This package embodies the dynamic strategy selection.

### `main.retrieval.indexing`: Responsible for creating spatial data structures for efficient searching.

- **KDTreeBuilder**: Constructs the K-D Tree.
- **KDNode**: Represents a node within the K-D Tree.

### `main.retrieval.search`: Contains the logic for performing similarity searches.

- **interfaces**: Defines the contracts for search strategies (`Searchable`, `Buildable`, `Insertable`).
- **implementations**: Provides concrete search strategies like `DeepMetricSearch` and `BestBinFirstSearch`.
- **annotations**: Includes custom annotations like `@SearchCapabilities` to provide metadata about the search strategies.

### `main.retrieval.models`: Defines the core data structures.

- **ImageFeature**: A simple POJO that encapsulates an image identifier and its corresponding numerical feature vector.

### `main.retrieval.utils`: A collection of utility classes.

- **FeatureUtils**: Provides static methods for mathematical operations on feature vectors, such as normalization and distance calculations.

## 4. Getting Started

### Prerequisites

- Java JDK 11 or higher
- Maven 3.6 or higher
- An active internet connection (for downloading Maven dependencies and pre-trained models on the first run).

### Installation & Setup

Clone the repository:

```bash
git clone [https://github.com/your-username/image-retrieval-system.git](https://github.com/your-username/image-retrieval-system.git)
cd image-retrieval-system
```

Build the project with Maven:

```bash
mvn clean install
```

This command will compile the source code, run any existing tests, and download all the required dependencies, including DJL, DL4J, OpenCV, and BoofCV.

## 5. Usage

The system is designed to be used as a library within a larger application. Here is a conceptual example of how to index a directory of images and then query it to find similar images.

### Step 1: Indexing a Collection of Images

```java
import main.java.com.retrieval.features.ExtractorFactory;
import main.java.com.retrieval.features.extractor.Extractable;
import main.java.com.retrieval.models.ImageFeature;
import main.java.com.retrieval.search.implementations.DeepMetricSearch;
import main.java.com.retrieval.search.interfaces.Buildable;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class IndexingExample {
    public static void main(String[] args) {
        File imageDirectory = new File("path/to/your/images");
        List<ImageFeature> featureList = new ArrayList<>();

        for (File imageFile : imageDirectory.listFiles()) {
            if (imageFile.isFile()) {
                String imagePath = imageFile.getAbsolutePath();
                // The factory dynamically selects the best extractor
                Extractable extractor = ExtractorFactory.getExtractor(imagePath);
                double[] featureVector = extractor.extract(imagePath);

                if (featureVector.length > 0) {
                    String imageId = imageFile.getName();
                    featureList.add(new ImageFeature(imageId, featureVector));
                }
            }
        }

        // Initialize and build the search index
        Buildable searchIndex = new DeepMetricSearch();
        searchIndex.buildIndex(featureList);

        System.out.println("Successfully indexed " + featureList.size() + " images.");
        // This searchIndex object can now be used for querying
    }
}
```

### Step 2: Querying for Similar Images

```java
import main.java.com.retrieval.features.ExtractorFactory;
import main.java.com.retrieval.features.extractor.Extractable;
import main.java.com.retrieval.models.ImageFeature;
import main.java.com.retrieval.search.interfaces.Searchable;

import java.util.List;

public class QueryExample {
    public static void main(String[] args) {
        // Assume 'searchIndex' is the built index from the previous step
        Searchable searcher = (Searchable) searchIndex; // Cast to the Searchable interface

        String queryImagePath = "path/to/query/image.jpg";
        int topK = 5; // Number of similar images to retrieve

        // Extract features for the query image using the same dynamic factory
        Extractable queryExtractor = ExtractorFactory.getExtractor(queryImagePath);
        double[] queryVector = queryExtractor.extract(queryImagePath);

        if (queryVector.length > 0) {
            List<ImageFeature> results = searcher.query(queryVector, topK);

            System.out.println("Top " + topK + " similar images:");
            for (ImageFeature result : results) {
                System.out.println(" - " + result.getImageId());
            }
        } else {
            System.out.println("Could not extract features from the query image.");
        }
    }
}
```

## 6. TODO: Testing and Future Work

This section outlines the plan for comprehensive testing and future enhancements to ensure the system's reliability, performance, and maintainability.

### I. Unit Testing (High Priority)

The goal is to achieve high test coverage for individual components, ensuring they behave as expected in isolation and handle edge cases gracefully.

#### `main.retrieval.utils.FeatureUtils`:

- `normalize`: Test with zero vectors, non-zero vectors, and already normalized vectors. Verify in-place modification.
- `normalizedCopy`: Verify it returns a new normalized vector and does not modify the original.
- `cosineDistance`: Test with identical vectors (distance 0), orthogonal vectors (distance 1), and opposite vectors (distance 2). Test with zero vectors and vectors of different lengths (expect `IllegalArgumentException`).
- `euclideanDistance`: Test with identical vectors (distance 0) and known vector pairs. Test for exceptions with invalid inputs.
- `manhattanDistance`: Test with identical vectors and known vector pairs.
- `isNormalized`: Test with vectors that are normalized, not normalized, and close to the tolerance threshold.
- `calculateStatistics`: Test with a sample vector and verify the calculated mean, stddev, min, and max.

#### `main.retrieval.features.extractor.*`:

For each extractor (`ORBExtractor`, `BoofCVExtractor`, `DJLExtractor`, `DL4JExtractor`):

- Test with a valid image path: ensure it returns a non-empty `double[]`.
- Test with a non-existent image path: ensure it returns an empty `double[]` and logs an appropriate error.
- Test with a corrupted/unsupported image file: ensure graceful failure (empty array) and logging.
- Test with a null or empty image path string: expect `IllegalArgumentException`.
- Verify that the output vector is L2 normalized.

#### `main.retrieval.features.rules.*`:

- `FileSizeRule`: Test with a file smaller than the threshold (expect `ORBExtractor`) and a file larger than the threshold (expect `null`).
- `ResolutionRule`: Test with a high-res image (expect `DJLExtractor`) and a low-res image (expect `null`).
- `MetadataCameraRule`: Create mock image files with and without EXIF camera model tags ("Canon", "Nikon", "Sony"). Verify it returns `DL4JExtractor` when the tag is present and null otherwise.
- `RuleEngine`: Mock the list of rules. Test the chain of responsibility: ensure the first rule that returns a non-null result is used. Test the fallback case where no rules match (expect `BoofCVExtractor`).

#### `main.retrieval.indexing.KDTreeBuilder`:

- Test with an empty list of features (expect null root).
- Test with a list of features having different dimensions (should ideally be handled before this stage, but good to test for robustness).
- Build a tree with a small, known set of 2D points and manually verify its structure (root, axis, left/right children).

#### `main.retrieval.search.implementations.*`:

**DeepMetricSearch**:

- `buildIndex`: Test building with an empty list and a populated list.
- `insert`: Test inserting features and verify the index size increases. Test thread safety by inserting from multiple threads concurrently.
- `query`: Test with `k > number of items` in index. Test with an empty index (expect `IllegalStateException` or empty list). Verify that results are sorted correctly by cosine distance.

**BestBinFirstSearch**:

- `buildIndex`: Test building with empty and populated lists.
- `query`: Test against a known K-D Tree structure. Verify that it returns `k` results. Test with `useCosineSimilarity` set to false (Euclidean). Test with invalid arguments (`k=0`, null query vector).

### II. Integration Testing

Integration tests will focus on the collaboration between different modules, primarily the pipeline from feature extraction to searching.

**Full Indexing and Querying Pipeline**:

Create a test that takes a small directory of diverse images (high-res, low-res, with/without EXIF).

The test will:

- Iterate through images, using `ExtractorFactory` to get features.
- Use `DeepMetricSearch` to build an index from these features.
- Pick one image as a query.
- Extract its features.
- Perform a `query(queryVector, k=1)`.
- Assert that the top result is the query image itself.
- Repeat the above pipeline using `BestBinFirstSearch`.

### III. Future Enhancements

**Performance Benchmarking**:

Create benchmark tests (e.g., using JMH) to measure:

- The time taken by each `Extractable` implementation for various image sizes.
- The time taken to build indexes (`DeepMetricSearch` vs. `BestBinFirstSearch`) for 1k, 10k, and 100k features.
- Query latency for both search implementations as the index size grows.

**Code Refactoring and Cleanup**:

- **Dependency Management**: Ensure all deep learning models (`DJLExtractor`, `DL4JExtractor`) properly close resources (`AutoCloseable`).
- **Configuration**: Externalize magic numbers (e.g., `maxChecks` in `BestBinFirstSearch`, image dimensions) into a configuration file (e.g., `application.properties`).

**Documentation**:

- Add detailed Javadoc comments to all public methods and classes.
- Create a more comprehensive `USAGE.md` with advanced examples, including how to add a new custom extractor or searcher.

**New Features**:

- **Index Persistence**: Add functionality to save a built index (e.g., the K-D Tree or the feature list) to disk and load it back, avoiding the need to re-process all images on every application start.
- **Alternative Indexing**: Implement more advanced Approximate Nearest Neighbor (ANN) indexing strategies.
