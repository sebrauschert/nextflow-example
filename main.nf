#!/usr/bin/env nextflow

/*
 * Nextflow workflow for reproducible data science
 * A containerized workflow for data preprocessing and model training
 */

nextflow.enable.dsl=2

// Parameters
params.input = "data/raw_data.csv"
params.outdir = "results"
params.seed = 42

// Log info
log.info """\
         REPRODUCIBLE DATA SCIENCE PIPELINE
         =================================
         Input data   : ${params.input}
         Output dir   : ${params.outdir}
         Random seed  : ${params.seed}
         """
         .stripIndent()

// Process for data preprocessing
process preprocess {
    container 'preprocess:latest'
    
    input:
    path input_file
    
    output:
    path "processed_data.csv"
    
    script:
    """
    python /app/scripts/preprocess.py ${input_file} processed_data.csv
    """
}

// Process for model training
process train_model {
    container 'model:latest'
    
    input:
    path processed_data
    val seed
    
    output:
    path "model.pkl"
    path "metrics.json"
    
    script:
    """
    python /app/scripts/train_model.py ${processed_data} ${seed} model.pkl metrics.json
    """
}

// Main workflow
workflow {
    // Create channel from input file
    input_ch = channel.fromPath(params.input, checkIfExists: true)
    
    // Run preprocessing
    processed_ch = preprocess(input_ch)
    
    // Train model and evaluate
    train_model(processed_ch, params.seed)
}

// Workflow completion handler
workflow.onComplete {
    log.info "Pipeline completed at: $workflow.complete"
    log.info "Execution status: ${ workflow.success ? 'SUCCESS' : 'FAILED' }"
    log.info "Execution duration: $workflow.duration"
}