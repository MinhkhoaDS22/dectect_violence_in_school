// lib/models/alert_model.dart
class ViolenceSegment {
  final double startSec;
  final double endSec;
  final double confidence;

  const ViolenceSegment({
    required this.startSec,
    required this.endSec,
    required this.confidence,
  });

  factory ViolenceSegment.fromJson(Map<String, dynamic> j) => ViolenceSegment(
        startSec: (j['start_sec'] as num).toDouble(),
        endSec: (j['end_sec'] as num).toDouble(),
        confidence: (j['confidence'] as num).toDouble(),
      );

  String get duration {
    final d = endSec - startSec;
    return '${d.toStringAsFixed(1)}s';
  }

  String get timeRange =>
      '${startSec.toStringAsFixed(1)}s → ${endSec.toStringAsFixed(1)}s';
}

class NotificationStatus {
  final String? email;
  final String? telegram;

  const NotificationStatus({this.email, this.telegram});

  factory NotificationStatus.fromJson(Map<String, dynamic> j) =>
      NotificationStatus(
        email: j['email'] as String?,
        telegram: j['telegram'] as String?,
      );

  bool get anySuccess =>
      email == 'sent' || telegram == 'sent';
}

class AlertModel {
  final String id;
  final DateTime timestamp;
  final String videoFilename;
  final String? email;
  final String? phone;
  final String? telegramChatId;
  final bool isViolence;
  final List<ViolenceSegment> segments;
  final double videoDuration;
  final double violenceRatio;
  final int maxViolentPersons;
  final String summary;
  final List<String> clips;
  final NotificationStatus notificationStatus;

  const AlertModel({
    required this.id,
    required this.timestamp,
    required this.videoFilename,
    this.email,
    this.phone,
    this.telegramChatId,
    required this.isViolence,
    required this.segments,
    required this.videoDuration,
    required this.violenceRatio,
    this.maxViolentPersons = 0,
    required this.summary,
    required this.clips,
    required this.notificationStatus,
  });

  factory AlertModel.fromJson(Map<String, dynamic> j) => AlertModel(
        id: j['id'] as String,
        timestamp: DateTime.parse(j['timestamp'] as String),
        videoFilename: j['video_filename'] as String,
        email: j['email'] as String?,
        phone: j['phone'] as String?,
        telegramChatId: j['telegram_chat_id'] as String?,
        isViolence: j['is_violence'] as bool,
        segments: (j['segments'] as List)
            .map((e) => ViolenceSegment.fromJson(e as Map<String, dynamic>))
            .toList(),
        videoDuration: (j['video_duration'] as num).toDouble(),
        violenceRatio: (j['violence_ratio'] as num).toDouble(),
        maxViolentPersons: (j['max_violent_persons'] as num?)?.toInt() ?? 0,
        summary: j['summary'] as String,
        clips: List<String>.from(j['clips'] as List),
        notificationStatus: NotificationStatus.fromJson(
          j['notification_status'] as Map<String, dynamic>,
        ),
      );
}
