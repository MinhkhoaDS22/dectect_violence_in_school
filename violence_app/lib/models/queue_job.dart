// lib/models/queue_job.dart
import 'dart:typed_data';
import 'package:flutter/material.dart';

enum JobStatus { waiting, uploading, analyzing, done, error }

class QueueJob {
  final String id;
  final String fileName;
  final Uint8List bytes;

  JobStatus status;
  double progress;
  // Use dynamic to avoid circular import with api_service.dart
  // Cast to AnalyzeResult when needed in widgets
  dynamic result;
  String? errorMsg;
  DateTime createdAt;

  QueueJob({
    required this.id,
    required this.fileName,
    required this.bytes,
    this.status = JobStatus.waiting,
    this.progress = 0,
    this.result,
    this.errorMsg,
    DateTime? createdAt,
  }) : createdAt = createdAt ?? DateTime.now();

  bool get isViolence => result?.isViolence == true;

  // ── Helpers ──────────────────────────────────────────────
  IconData get icon => switch (status) {
        JobStatus.waiting   => Icons.schedule_rounded,
        JobStatus.uploading => Icons.cloud_upload_rounded,
        JobStatus.analyzing => Icons.psychology_rounded,
        JobStatus.done      => isViolence
            ? Icons.warning_amber_rounded
            : Icons.check_circle_rounded,
        JobStatus.error     => Icons.error_outline_rounded,
      };

  Color get color => switch (status) {
        JobStatus.waiting   => const Color(0xFF8E8E93),
        JobStatus.uploading => const Color(0xFF0A84FF),
        JobStatus.analyzing => const Color(0xFFFF9F0A),
        JobStatus.done      => isViolence
            ? const Color(0xFFFF453A)
            : const Color(0xFF30D158),
        JobStatus.error     => const Color(0xFFFF453A),
      };

  String get statusLabel => switch (status) {
        JobStatus.waiting   => 'Đang chờ',
        JobStatus.uploading => 'Đang tải lên (${(progress * 100).toInt()}%)',
        JobStatus.analyzing => 'Đang phân tích...',
        JobStatus.done      => isViolence
            ? '🚨 Phát hiện bạo lực'
            : '✅ Bình thường',
        JobStatus.error     => 'Lỗi',
      };

  bool get isRunning =>
      status == JobStatus.uploading || status == JobStatus.analyzing;
  bool get isFinished => status == JobStatus.done || status == JobStatus.error;
}
