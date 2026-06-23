// lib/services/api_service.dart
//
// ApiService — đọc baseUrl từ SharedPreferences.
// Mặc định: http://localhost:8000
// User có thể đổi IP/domain trong Settings → lưu key 'backend_url'

import 'package:dio/dio.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/alert_model.dart';

class ApiService {
  static const _defaultUrl = 'http://localhost:8000';
  static const _prefKey    = 'backend_url';

  // Sync cache — được set mỗi khi getBaseUrl() được gọi
  static String _cachedBaseUrl = _defaultUrl;

  // ── Lấy base URL từ prefs (async) ─────────────────────────
  static Future<String> getBaseUrl() async {
    final p = await SharedPreferences.getInstance();
    final url = p.getString(_prefKey) ?? _defaultUrl;
    final clean = url.endsWith('/') ? url.substring(0, url.length - 1) : url;
    _cachedBaseUrl = clean; // update cache
    return clean;
  }

  // ── Sync URL (dùng cache, đọc được ngay trong build) ──────
  static String get baseUrlSync => _cachedBaseUrl;

  // ── Tạo Dio với baseUrl hiện tại ──────────────────────────
  Future<Dio> _dio() async {
    final base = await getBaseUrl();
    return Dio(BaseOptions(
      baseUrl: base,
      connectTimeout: const Duration(seconds: 15),
      receiveTimeout: const Duration(minutes: 10),
    ));
  }

  // ── Health check ──────────────────────────────────────────
  Future<bool> checkHealth() async {
    try {
      final dio = await _dio();
      final res = await dio.get('/api/health');
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  // ── Analyze video ─────────────────────────────────────────
  /// Uploads video and contact info, returns analysis result.
  /// [onProgress] called with 0.0..1.0 during upload
  Future<AnalyzeResult> analyzeVideo({
    required String fileName,
    required List<int> fileBytes,
    String? email,
    String? phone,
    String? telegramChatId,
    void Function(double)? onProgress,
  }) async {
    final dio = await _dio();
    final formData = FormData.fromMap({
      'video': MultipartFile.fromBytes(fileBytes, filename: fileName),
      if (email != null && email.isNotEmpty) 'email': email,
      if (phone != null && phone.isNotEmpty) 'phone': phone,
      if (telegramChatId != null && telegramChatId.isNotEmpty)
        'telegram_chat_id': telegramChatId,
    });

    final res = await dio.post(
      '/api/analyze',
      data: formData,
      onSendProgress: (sent, total) {
        if (total > 0) onProgress?.call(sent / total);
      },
    );

    return AnalyzeResult.fromJson(res.data as Map<String, dynamic>);
  }

  // ── Alert history ─────────────────────────────────────────
  Future<List<AlertModel>> getAlerts({int limit = 50, int offset = 0}) async {
    final dio = await _dio();
    final res = await dio.get('/api/alerts',
        queryParameters: {'limit': limit, 'offset': offset});
    final list = (res.data['alerts'] as List);
    return list
        .map((e) => AlertModel.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  Future<void> deleteAlert(String alertId) async {
    final dio = await _dio();
    await dio.delete('/api/alerts/$alertId');
  }

  Future<String> clipUrl(String jobId, String filename) async {
    final base = await getBaseUrl();
    return '$base/api/clips/$jobId/$filename';
  }

  /// Sync version — dùng cached URL, an toàn trong build()
  String clipUrlSync(String jobId, String filename) =>
      '$_cachedBaseUrl/api/clips/$jobId/$filename';
}

// ── Result wrapper ────────────────────────────────────────
class AnalyzeResult {
  final String jobId;
  final bool isViolence;
  final List<ViolenceSegment> segments;
  final double videoDuration;
  final double violenceRatio;
  final int maxViolentPersons;
  final String summary;
  final List<String> clips;
  final NotificationStatus notificationStatus;

  const AnalyzeResult({
    required this.jobId,
    required this.isViolence,
    required this.segments,
    required this.videoDuration,
    required this.violenceRatio,
    this.maxViolentPersons = 0,
    required this.summary,
    required this.clips,
    required this.notificationStatus,
  });

  factory AnalyzeResult.fromJson(Map<String, dynamic> j) {
    final result = j['result'] as Map<String, dynamic>;
    return AnalyzeResult(
      jobId: j['job_id'] as String,
      isViolence: result['is_violence'] as bool,
      segments: (result['segments'] as List)
          .map((e) => ViolenceSegment.fromJson(e as Map<String, dynamic>))
          .toList(),
      videoDuration: (result['video_duration'] as num).toDouble(),
      violenceRatio: (result['violence_ratio'] as num).toDouble(),
      maxViolentPersons: (result['max_violent_persons'] as num?)?.toInt() ?? 0,
      summary: result['summary'] as String,
      clips: List<String>.from(j['clips'] as List),
      notificationStatus: NotificationStatus.fromJson(
        j['notification_status'] as Map<String, dynamic>,
      ),
    );
  }
}
