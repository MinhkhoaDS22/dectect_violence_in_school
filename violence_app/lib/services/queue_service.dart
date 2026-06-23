// lib/services/queue_service.dart
//
// QueueService — xử lý hàng chờ video tuần tự.
// Giải pháp "nhiều camera cùng lúc" không cần Redis/Celery:
//   • Mỗi video là 1 QueueJob được thêm vào danh sách.
//   • Worker loop: khi job trước xong mới lấy job tiếp theo và gọi API.
//   • Future: thay _processNext bằng N concurrent worker để scale.

import 'package:flutter/foundation.dart';
import '../models/queue_job.dart';
import 'api_service.dart';
import 'sound_service.dart';

// Contact info per job (avoids modifying QueueJob model)
final Map<String, Map<String, String?>> _contactMap = {};

void attachContactInfo(
  String jobId, {
  String? email,
  String? phone,
  String? telegram,
}) {
  _contactMap[jobId] = {
    'email': email,
    'phone': phone,
    'telegram': telegram,
  };
}

class QueueService extends ChangeNotifier {
  // ── Singleton ─────────────────────────────────────────────
  static final QueueService instance = QueueService._();
  QueueService._();

  final _api = ApiService();
  final List<QueueJob> _jobs = [];
  bool _running = false;

  // ── Public state ──────────────────────────────────────────
  List<QueueJob> get jobs => List.unmodifiable(_jobs);

  int get waiting   => _jobs.where((j) => j.status == JobStatus.waiting).length;
  int get running   => _jobs.where((j) => j.isRunning).length;
  int get completed => _jobs.where((j) => j.isFinished).length;
  int get violenceCount =>
      _jobs.where((j) => j.isViolence).length;

  // ── Add job ───────────────────────────────────────────────
  void addJob(QueueJob job) {
    _jobs.add(job);
    notifyListeners();
    _startWorker();
  }

  // ── Remove / clear ────────────────────────────────────────
  void removeJob(String id) {
    _jobs.removeWhere((j) => j.id == id && j.isFinished);
    _contactMap.remove(id);
    notifyListeners();
  }

  void clearFinished() {
    final finishedIds = _jobs.where((j) => j.isFinished).map((j) => j.id).toList();
    _jobs.removeWhere((j) => j.isFinished);
    for (final id in finishedIds) { _contactMap.remove(id); }
    notifyListeners();
  }

  // ── Worker loop ───────────────────────────────────────────
  void _startWorker() {
    if (_running) return;
    _running = true;
    _processNext();
  }

  Future<void> _processNext() async {
    final job = _jobs.cast<QueueJob?>().firstWhere(
          (j) => j!.status == JobStatus.waiting,
          orElse: () => null,
        );

    if (job == null) {
      _running = false;
      return;
    }

    await _processJob(job);
    _processNext();
  }

  Future<void> _processJob(QueueJob job) async {
    final contact = _contactMap[job.id];

    _setStatus(job, JobStatus.uploading);

    try {
      final result = await _api.analyzeVideo(
        fileName: job.fileName,
        fileBytes: job.bytes,
        email: contact?['email'],
        phone: contact?['phone'],
        telegramChatId: contact?['telegram'],
        onProgress: (p) {
          job.progress = p;
          if (p >= 1.0) _setStatus(job, JobStatus.analyzing);
          notifyListeners();
        },
      );

      job.result = result;
      _setStatus(job, JobStatus.done);

      if (result.isViolence) {
        SoundService.instance.playAlert();
      }
    } catch (e) {
      job.errorMsg = e.toString();
      _setStatus(job, JobStatus.error);
    }
  }

  void _setStatus(QueueJob job, JobStatus status) {
    job.status = status;
    notifyListeners();
  }
}
