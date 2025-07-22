package com.retrieval.features.rules;

import com.retrieval.features.extractor.Extractable;

import java.io.File;

public interface ExtractorRule {
    Extractable apply (File imageFile);
}
