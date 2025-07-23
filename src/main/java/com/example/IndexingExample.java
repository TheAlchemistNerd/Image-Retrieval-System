package com.example;

import com.retrieval.features.extractor.Extractable;
import com.retrieval.features.ExtractorFactory;
import com.retrieval.models.ImageFeature;
import com.retrieval.search.implementations.DeepMetricSearch;
import com.retrieval.search.interfaces.Buildable;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class IndexingExample {
    public static void main(String[] args) {
        File imageDirectory = new File("C:\\Users\\Nevo\\Documents\\Projects\\Shoes - Pics");
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