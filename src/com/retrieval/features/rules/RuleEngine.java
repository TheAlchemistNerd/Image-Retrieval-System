package com.retrieval.features.rules;

import com.retrieval.features.extractor.BoofCVExtractor;
import com.retrieval.features.extractor.Extractable;

import java.io.File;
import java.util.List;

public class RuleEngine {
    public static final List<ExtractorRule> RULES = List.of(
            new FileSizeRule(),
            new MetadataCameraRule(),
            new ResolutionRule()
    );

    public static Extractable runRules(File file) {
        for (ExtractorRule rule : RULES) {
            Extractable result = rule.apply(file);
            if(result != null) return result;
        }
        return new BoofCVExtractor();
    }
}
