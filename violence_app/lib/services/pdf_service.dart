// lib/services/pdf_service.dart
//
// Tạo báo cáo PDF từ kết quả phân tích bạo lực.
// Sử dụng package `pdf` + `printing` để xuất file cho Flutter Web.

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';
import '../services/api_service.dart';

class PdfService {
  static Future<void> exportAnalysisResult({
    required BuildContext context,
    required AnalyzeResult result,
    required String fileName,
  }) async {
    final pdf = pw.Document(
      title: 'SafeWatch — Báo cáo phân tích',
      author: 'SafeWatch Violence Detection System',
    );

    final now = DateFormat('dd/MM/yyyy HH:mm:ss').format(DateTime.now());
    final isVio = result.isViolence;

    pdf.addPage(
      pw.MultiPage(
        pageFormat: PdfPageFormat.a4,
        margin: const pw.EdgeInsets.all(40),
        header: (_) => _buildHeader(fileName, now),
        footer: (ctx) => _buildFooter(ctx),
        build: (ctx) => [
          pw.SizedBox(height: 24),

          // Result banner
          pw.Container(
            width: double.infinity,
            padding: const pw.EdgeInsets.all(20),
            decoration: pw.BoxDecoration(
              color: isVio
                  ? PdfColor.fromHex('FFF0F0')
                  : PdfColor.fromHex('F0FFF4'),
              borderRadius: pw.BorderRadius.circular(8),
              border: pw.Border.all(
                color: isVio
                    ? PdfColor.fromHex('FF453A')
                    : PdfColor.fromHex('30D158'),
                width: 1,
              ),
            ),
            child: pw.Column(
              crossAxisAlignment: pw.CrossAxisAlignment.start,
              children: [
                pw.Text(
                  isVio ? '🚨 PHÁT HIỆN BẠO LỰC' : '✅ VIDEO BÌNH THƯỜNG',
                  style: pw.TextStyle(
                    fontSize: 18,
                    fontWeight: pw.FontWeight.bold,
                    color: isVio
                        ? PdfColor.fromHex('FF453A')
                        : PdfColor.fromHex('30D158'),
                  ),
                ),
                pw.SizedBox(height: 8),
                pw.Text(result.summary,
                    style: const pw.TextStyle(fontSize: 12)),
              ],
            ),
          ),

          pw.SizedBox(height: 20),

          // Stats table
          pw.Text('Thống kê phân tích',
              style: pw.TextStyle(
                  fontSize: 14, fontWeight: pw.FontWeight.bold)),
          pw.SizedBox(height: 10),
          pw.TableHelper.fromTextArray(
            headers: ['Chỉ số', 'Giá trị'],
            data: [
              ['File video', fileName],
              ['Thời lượng', '${result.videoDuration.toStringAsFixed(1)}s'],
              [
                'Tỷ lệ bạo lực',
                '${(result.violenceRatio * 100).toStringAsFixed(1)}%'
              ],
              ['Số người BL tối đa', '${result.maxViolentPersons}'],
              ['Số đoạn bạo lực', '${result.segments.length}'],
              ['Clip đã cắt', '${result.clips.length}'],
              ['Thời điểm phân tích', now],
            ],
            headerStyle: pw.TextStyle(
                fontWeight: pw.FontWeight.bold, color: PdfColors.white),
            headerDecoration:
                const pw.BoxDecoration(color: PdfColor.fromInt(0xFF0A84FF)),
            cellStyle: const pw.TextStyle(fontSize: 11),
            cellAlignments: {
              0: pw.Alignment.centerLeft,
              1: pw.Alignment.centerLeft,
            },
            border: pw.TableBorder.all(color: PdfColors.grey300),
          ),

          if (result.segments.isNotEmpty) ...[
            pw.SizedBox(height: 20),
            pw.Text('Các đoạn bạo lực phát hiện',
                style: pw.TextStyle(
                    fontSize: 14, fontWeight: pw.FontWeight.bold)),
            pw.SizedBox(height: 10),
            pw.TableHelper.fromTextArray(
              headers: ['#', 'Bắt đầu', 'Kết thúc', 'Dài', 'Độ tin cậy'],
              data: result.segments.asMap().entries.map((e) {
                final i = e.key;
                final s = e.value;
                return [
                  '${i + 1}',
                  '${s.startSec.toStringAsFixed(1)}s',
                  '${s.endSec.toStringAsFixed(1)}s',
                  s.duration,
                  '${s.confidence.toStringAsFixed(1)}%',
                ];
              }).toList(),
              headerStyle: pw.TextStyle(
                  fontWeight: pw.FontWeight.bold, color: PdfColors.white),
              headerDecoration:
                  const pw.BoxDecoration(color: PdfColor.fromInt(0xFFFF453A)),
              cellStyle: const pw.TextStyle(fontSize: 11),
              border: pw.TableBorder.all(color: PdfColors.grey300),
            ),
          ],

          pw.SizedBox(height: 20),
          _buildDisclaimer(),
        ],
      ),
    );

    await Printing.sharePdf(
      bytes: await pdf.save(),
      filename:
          'safwatch_report_${DateTime.now().millisecondsSinceEpoch}.pdf',
    );
  }

  static pw.Widget _buildHeader(String fileName, String date) =>
      pw.Column(children: [
        pw.Row(
          mainAxisAlignment: pw.MainAxisAlignment.spaceBetween,
          children: [
            pw.Column(crossAxisAlignment: pw.CrossAxisAlignment.start, children: [
              pw.Text('SafeWatch',
                  style: pw.TextStyle(
                      fontSize: 20,
                      fontWeight: pw.FontWeight.bold,
                      color: PdfColor.fromHex('0A84FF'))),
              pw.Text('Hệ thống phát hiện bạo lực học đường',
                  style: const pw.TextStyle(
                      fontSize: 10, color: PdfColors.grey600)),
            ]),
            pw.Column(crossAxisAlignment: pw.CrossAxisAlignment.end, children: [
              pw.Text('Báo cáo phân tích',
                  style: pw.TextStyle(
                      fontSize: 12, fontWeight: pw.FontWeight.bold)),
              pw.Text(date,
                  style: const pw.TextStyle(
                      fontSize: 10, color: PdfColors.grey600)),
            ]),
          ],
        ),
        pw.Divider(color: PdfColor.fromHex('0A84FF'), thickness: 1.5),
        pw.SizedBox(height: 4),
      ]);

  static pw.Widget _buildFooter(pw.Context ctx) => pw.Column(children: [
        pw.Divider(color: PdfColors.grey300),
        pw.Row(
          mainAxisAlignment: pw.MainAxisAlignment.spaceBetween,
          children: [
            pw.Text('SafeWatch — Phát hiện bạo lực học đường (HUTECH 2025)',
                style: const pw.TextStyle(
                    fontSize: 9, color: PdfColors.grey500)),
            pw.Text('Trang ${ctx.pageNumber}/${ctx.pagesCount}',
                style: const pw.TextStyle(
                    fontSize: 9, color: PdfColors.grey500)),
          ],
        ),
      ]);

  static pw.Widget _buildDisclaimer() => pw.Container(
        padding: const pw.EdgeInsets.all(12),
        decoration: pw.BoxDecoration(
          color: PdfColors.amber50,
          borderRadius: pw.BorderRadius.circular(6),
          border: pw.Border.all(color: PdfColors.amber200),
        ),
        child: pw.Column(
          crossAxisAlignment: pw.CrossAxisAlignment.start,
          children: [
            pw.Text('Lưu ý',
                style: pw.TextStyle(
                    fontSize: 11, fontWeight: pw.FontWeight.bold)),
            pw.SizedBox(height: 4),
            pw.Text(
              'Kết quả phân tích được tạo tự động bởi mô hình AI (CNN-BiLSTM-Attention). '
              'Hệ thống có thể có sai số. Kết quả cần được xác nhận bởi người có thẩm quyền '
              'trước khi đưa ra quyết định. Báo cáo này chỉ phục vụ mục đích tham khảo.',
              style: const pw.TextStyle(fontSize: 10, color: PdfColors.grey700),
            ),
          ],
        ),
      );
}
