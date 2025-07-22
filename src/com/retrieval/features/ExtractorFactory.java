package com.retrieval.features;

import com.retrieval.features.extractor.Extractable;
import com.retrieval.features.extractor.ORBExtractor;
import com.retrieval.features.rules.RuleEngine;

import java.io.File;
import java.util.logging.Logger;

public class ExtractorFactory {

    private static final Logger LOGGER = Logger.getLogger(ExtractorFactory.class.getName());

    public static Extractable getExtractor(String imagePath) {
        File imageFile = new File(imagePath);
        if (!imageFile.exists()) {
            LOGGER.warning("File does not exist. Returning ORBExtractor fallback.");
            return new ORBExtractor();
        }

        return RuleEngine.runRules(imageFile);
    }
}
