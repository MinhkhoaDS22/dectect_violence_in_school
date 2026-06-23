// lib/widgets/violence_timeline.dart
//
// ViolenceTimeline — thanh timeline trực quan hóa các đoạn bạo lực.
// Màu đỏ = bạo lực, màu xanh = bình thường.
// Hover/tap vào đoạn đỏ để xem thời gian và độ tin cậy.

import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../models/alert_model.dart';
import '../theme/app_theme.dart';

class ViolenceTimeline extends StatefulWidget {
  final double videoDuration;
  final List<ViolenceSegment> segments;
  final bool showLabel;

  const ViolenceTimeline({
    super.key,
    required this.videoDuration,
    required this.segments,
    this.showLabel = true,
  });

  @override
  State<ViolenceTimeline> createState() => _ViolenceTimelineState();
}

class _ViolenceTimelineState extends State<ViolenceTimeline> {
  int? _hoveredIndex;

  @override
  Widget build(BuildContext context) {
    final dur = widget.videoDuration;
    if (dur <= 0) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (widget.showLabel)
          Row(
            children: [
              const Icon(Icons.timeline_rounded,
                  size: 16, color: AppColors.textSecondary),
              const SizedBox(width: 6),
              Text(
                'Timeline phân tích video',
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
              ),
              const Spacer(),
              _LegendDot(color: AppColors.success, label: 'Bình thường'),
              const SizedBox(width: 12),
              _LegendDot(color: AppColors.danger, label: 'Bạo lực'),
            ],
          ),
        if (widget.showLabel) const SizedBox(height: 10),

        // ── Timeline bar ──────────────────────────────────────
        LayoutBuilder(builder: (context, constraints) {
          final totalWidth = constraints.maxWidth;
          return Column(
            children: [
              // Bar
              ClipRRect(
                borderRadius: BorderRadius.circular(6),
                child: SizedBox(
                  height: 20,
                  child: Stack(
                    children: [
                      // Background (safe)
                      Container(color: AppColors.successSoft),

                      // Violence segments
                      ...widget.segments.asMap().entries.map((e) {
                        final i = e.key;
                        final seg = e.value;
                        final left = (seg.startSec / dur) * totalWidth;
                        final width =
                            ((seg.endSec - seg.startSec) / dur) * totalWidth;
                        final isHovered = _hoveredIndex == i;

                        return Positioned(
                          left: left,
                          width: width.clamp(2.0, totalWidth),
                          top: 0,
                          bottom: 0,
                          child: MouseRegion(
                            onEnter: (_) =>
                                setState(() => _hoveredIndex = i),
                            onExit: (_) =>
                                setState(() => _hoveredIndex = null),
                            child: AnimatedContainer(
                              duration: 150.ms,
                              color: isHovered
                                  ? AppColors.danger
                                  : AppColors.danger.withOpacity(.8),
                              child: isHovered && width > 30
                                  ? Center(
                                      child: Text(
                                        '${seg.confidence.toStringAsFixed(0)}%',
                                        style: const TextStyle(
                                          color: Colors.white,
                                          fontSize: 10,
                                          fontWeight: FontWeight.w700,
                                        ),
                                      ),
                                    )
                                  : null,
                            ),
                          ),
                        );
                      }),
                    ],
                  ),
                ),
              ).animate().scaleX(
                    begin: 0,
                    end: 1,
                    alignment: Alignment.centerLeft,
                    duration: 600.ms,
                    curve: Curves.easeOut,
                  ),

              const SizedBox(height: 4),

              // Time labels
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text('0s',
                      style: Theme.of(context)
                          .textTheme
                          .bodyMedium
                          ?.copyWith(fontSize: 10)),
                  Text('${(dur / 2).toStringAsFixed(0)}s',
                      style: Theme.of(context)
                          .textTheme
                          .bodyMedium
                          ?.copyWith(fontSize: 10)),
                  Text('${dur.toStringAsFixed(0)}s',
                      style: Theme.of(context)
                          .textTheme
                          .bodyMedium
                          ?.copyWith(fontSize: 10)),
                ],
              ),
            ],
          );
        }),

        // ── Hovered segment tooltip ───────────────────────────
        if (_hoveredIndex != null &&
            _hoveredIndex! < widget.segments.length) ...[
          const SizedBox(height: 8),
          AnimatedOpacity(
            opacity: 1,
            duration: 150.ms,
            child: Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: AppColors.dangerSoft,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                    color: AppColors.danger.withOpacity(.3)),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.warning_amber_rounded,
                      size: 14, color: AppColors.danger),
                  const SizedBox(width: 6),
                  Text(
                    'Đoạn ${_hoveredIndex! + 1}: '
                    '${widget.segments[_hoveredIndex!].timeRange}  '
                    '• Độ tin cậy: '
                    '${widget.segments[_hoveredIndex!].confidence.toStringAsFixed(1)}%',
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                          color: AppColors.danger,
                          fontSize: 12,
                        ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ],
    );
  }
}

class _LegendDot extends StatelessWidget {
  final Color color;
  final String label;
  const _LegendDot({required this.color, required this.label});

  @override
  Widget build(BuildContext context) => Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 10,
            height: 10,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          const SizedBox(width: 4),
          Text(label,
              style: Theme.of(context)
                  .textTheme
                  .bodyMedium
                  ?.copyWith(fontSize: 10)),
        ],
      );
}
