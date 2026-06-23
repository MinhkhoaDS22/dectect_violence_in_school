// lib/screens/settings_screen.dart
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../theme/app_theme.dart';
import '../services/api_service.dart';
import '../widgets/sound_control_bar.dart';
import 'onboarding_screen.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _emailCtrl    = TextEditingController();
  final _phoneCtrl    = TextEditingController();
  final _telegramCtrl = TextEditingController();
  final _serverCtrl   = TextEditingController();
  final _formKey = GlobalKey<FormState>();
  bool _saving = false;
  bool _testingConnection = false;
  String? _connectionStatus;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _emailCtrl.dispose();
    _phoneCtrl.dispose();
    _telegramCtrl.dispose();
    _serverCtrl.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    final p = await SharedPreferences.getInstance();
    _emailCtrl.text    = p.getString('email') ?? '';
    _phoneCtrl.text    = p.getString('phone') ?? '';
    _telegramCtrl.text = p.getString('telegram_chat_id') ?? '';
    _serverCtrl.text   = p.getString('backend_url') ?? 'http://localhost:8000';
    if (mounted) setState(() {});
  }

  Future<void> _save() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _saving = true);
    final p = await SharedPreferences.getInstance();
    await p.setString('email',           _emailCtrl.text.trim());
    await p.setString('phone',           _phoneCtrl.text.trim());
    await p.setString('telegram_chat_id', _telegramCtrl.text.trim());

    // Lưu backend URL (bỏ trailing slash)
    var url = _serverCtrl.text.trim();
    if (url.endsWith('/')) url = url.substring(0, url.length - 1);
    await p.setString('backend_url', url);

    setState(() => _saving = false);
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: const Text('Đã lưu cài đặt'),
        backgroundColor: AppColors.success,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    );
  }

  Future<void> _testConnection() async {
    if (_serverCtrl.text.trim().isEmpty) return;
    setState(() {
      _testingConnection = true;
      _connectionStatus = null;
    });

    // Tạm thời lưu URL để test
    final p = await SharedPreferences.getInstance();
    await p.setString('backend_url', _serverCtrl.text.trim());

    final ok = await ApiService().checkHealth();
    if (mounted) {
      setState(() {
        _testingConnection = false;
        _connectionStatus = ok ? 'ok' : 'fail';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 480),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Header
                Row(
                  children: [
                    Container(
                      width: 48, height: 48,
                      decoration: BoxDecoration(
                        color: AppColors.primary.withOpacity(.12),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Icon(Icons.settings_rounded,
                          color: AppColors.primary, size: 24),
                    ),
                    const SizedBox(width: 16),
                    Text('Cài đặt',
                        style: Theme.of(context).textTheme.displayMedium),
                  ],
                ),
                const SizedBox(height: 32),

                // ── Backend Server URL ────────────────────────
                _SectionTitle('Kết nối Server'),
                const SizedBox(height: 12),

                // URL input + Test button
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      child: TextFormField(
                        controller: _serverCtrl,
                        decoration: const InputDecoration(
                          labelText: 'Backend URL',
                          hintText: 'http://192.168.1.x:8000',
                          prefixIcon: Icon(Icons.dns_rounded, size: 18),
                          helperText: 'IP hoặc domain của máy chạy FastAPI',
                        ),
                        validator: (v) {
                          if (v == null || v.trim().isEmpty) {
                            return 'Vui lòng nhập URL server';
                          }
                          if (!v.trim().startsWith('http')) {
                            return 'URL phải bắt đầu bằng http:// hoặc https://';
                          }
                          return null;
                        },
                        onChanged: (_) => setState(() => _connectionStatus = null),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: _testingConnection
                          ? const SizedBox(
                              width: 44, height: 44,
                              child: Center(
                                child: SizedBox(
                                  width: 20, height: 20,
                                  child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                      color: AppColors.primary),
                                ),
                              ),
                            )
                          : IconButton.filled(
                              onPressed: _testConnection,
                              icon: const Icon(Icons.network_check_rounded),
                              tooltip: 'Kiểm tra kết nối',
                              style: IconButton.styleFrom(
                                backgroundColor:
                                    AppColors.primary.withOpacity(.12),
                                foregroundColor: AppColors.primary,
                              ),
                            ),
                    ),
                  ],
                ),

                // Connection status badge
                if (_connectionStatus != null) ...[
                  const SizedBox(height: 8),
                  AnimatedContainer(
                    duration: const Duration(milliseconds: 200),
                    padding: const EdgeInsets.symmetric(
                        horizontal: 12, vertical: 8),
                    decoration: BoxDecoration(
                      color: _connectionStatus == 'ok'
                          ? AppColors.successSoft
                          : AppColors.dangerSoft,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          _connectionStatus == 'ok'
                              ? Icons.check_circle_rounded
                              : Icons.cancel_rounded,
                          size: 16,
                          color: _connectionStatus == 'ok'
                              ? AppColors.success
                              : AppColors.danger,
                        ),
                        const SizedBox(width: 6),
                        Text(
                          _connectionStatus == 'ok'
                              ? 'Kết nối thành công!'
                              : 'Không kết nối được — kiểm tra lại URL',
                          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                                color: _connectionStatus == 'ok'
                                    ? AppColors.success
                                    : AppColors.danger,
                                fontWeight: FontWeight.w600,
                                fontSize: 12,
                              ),
                        ),
                      ],
                    ),
                  ),
                ],

                const SizedBox(height: 32),
                const Divider(),
                const SizedBox(height: 24),

                // ── Thông tin liên hệ ─────────────────────────
                _SectionTitle('Thông tin liên hệ'),
                const SizedBox(height: 16),

                _SettingsField(
                  label: 'Gmail',
                  hint: 'example@gmail.com',
                  icon: Icons.email_outlined,
                  controller: _emailCtrl,
                  validator: (v) {
                    if (v != null && v.isNotEmpty) {
                      if (!RegExp(r'^[\w.+-]+@[\w-]+\.\w+$').hasMatch(v)) {
                        return 'Email không hợp lệ';
                      }
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 16),
                _SettingsField(
                  label: 'Số điện thoại',
                  hint: '0901234567',
                  icon: Icons.phone_outlined,
                  controller: _phoneCtrl,
                  keyboardType: TextInputType.phone,
                ),
                const SizedBox(height: 16),
                _SettingsField(
                  label: 'Telegram Chat ID',
                  hint: '123456789',
                  icon: Icons.telegram,
                  controller: _telegramCtrl,
                  keyboardType: TextInputType.number,
                  helperText: 'Nhắn /start cho bot → lấy chat_id từ @userinfobot',
                ),
                const SizedBox(height: 32),

                // Save button
                ElevatedButton.icon(
                  onPressed: _saving ? null : _save,
                  icon: _saving
                      ? const SizedBox(
                          width: 16, height: 16,
                          child: CircularProgressIndicator(
                              strokeWidth: 2, color: Colors.white))
                      : const Icon(Icons.save_rounded),
                  label: const Text('Lưu cài đặt'),
                ),

                const SizedBox(height: 32),
                const Divider(),
                const SizedBox(height: 24),

                // ── Âm thanh ────────────────────────────────
                _SectionTitle('Âm thanh cảnh báo'),
                const SizedBox(height: 12),
                const SoundControlBar(),

                const SizedBox(height: 32),
                const Divider(),
                const SizedBox(height: 24),

                // ── Hệ thống ──────────────────────────────────
                _SectionTitle('Hệ thống'),
                const SizedBox(height: 16),

                _InfoRow(label: 'Model',   value: 'CNN-BiLSTM-Attention'),
                _InfoRow(label: 'YOLO',    value: 'YOLOv11s'),
                _InfoRow(label: 'Phiên bản', value: '1.0.0'),

                const SizedBox(height: 32),
                OutlinedButton.icon(
                  onPressed: () => Navigator.pushReplacement(
                    context,
                    MaterialPageRoute(
                        builder: (_) => const OnboardingScreen()),
                  ),
                  icon: const Icon(Icons.restart_alt_rounded),
                  label: const Text('Chạy lại Onboarding'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: AppColors.textSecondary,
                    side: const BorderSide(color: AppColors.divider),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ── Helper widgets ─────────────────────────────────────────
class _SectionTitle extends StatelessWidget {
  final String text;
  const _SectionTitle(this.text);
  @override
  Widget build(BuildContext context) => Text(
        text,
        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              color: AppColors.textTertiary,
              fontWeight: FontWeight.w600,
              fontSize: 11,
              letterSpacing: .5,
            ),
      );
}

class _SettingsField extends StatelessWidget {
  final String label, hint;
  final IconData icon;
  final TextEditingController controller;
  final TextInputType? keyboardType;
  final String? Function(String?)? validator;
  final String? helperText;

  const _SettingsField({
    required this.label,
    required this.hint,
    required this.icon,
    required this.controller,
    this.keyboardType,
    this.validator,
    this.helperText,
  });

  @override
  Widget build(BuildContext context) => TextFormField(
        controller: controller,
        keyboardType: keyboardType,
        validator: validator,
        decoration: InputDecoration(
          labelText: label,
          hintText: hint,
          prefixIcon: Icon(icon, size: 18),
          helperText: helperText,
        ),
      );
}

class _InfoRow extends StatelessWidget {
  final String label, value;
  const _InfoRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Text(label, style: Theme.of(context).textTheme.bodyLarge),
          const Spacer(),
          Text(value,
              style: Theme.of(context)
                  .textTheme
                  .bodyMedium
                  ?.copyWith(color: AppColors.textTertiary)),
        ],
      ),
    );
  }
}
