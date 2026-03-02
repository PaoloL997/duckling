from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption


def warmup():
    pipeline_options = PdfPipelineOptions(
        generate_picture_images=True, do_formula_enrichment=True, images_scale=4
    )
    # accel_opts = AcceleratorOptions(device=AcceleratorDevice.CUDA, num_threads=8)
    # pipeline_options.accelerator_options = accel_opts
    pipeline_options.do_ocr = True
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    print(converter.convert(source="2408_09869v5.pdf").document)


if __name__ == "__main__":
    warmup()
