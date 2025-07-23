package com.retrieval.features.rules;

import com.retrieval.features.extractor.Extractable;
import com.retrieval.features.extractor.ORBExtractor;

import java.io.File;

public class FileSizeRule implements ExtractorRule{
    private static final long SMALL_FILE_THRESHOLD = Long.parseLong(System.getenv().getOrDefault("SMALL_FILE_SIZE_BYTES", "102400"));

    @Override
    public Extractable apply(File file) {
        return (file.length() < SMALL_FILE_THRESHOLD) ? new ORBExtractor() : null;
    }
}
