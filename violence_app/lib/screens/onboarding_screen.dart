// lib/screens/onboarding_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../theme/app_theme.dart';
import '../services/api_service.dart';
import 'home_screen.dart';

class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key});

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final _pageController = PageController();
  int _page = 0;

  // Form
  final _emailCtrl = TextEditingController();
  final _phoneCtrl = TextEditingController();
  final _telegramCtrl = TextEditingController();
  final _formKey = GlobalKey<FormState>();
  bool _saving = false;
  bool _checkingServer = false;
  bool? _serverOk;

  final _api = ApiService();

  @override
  void dispose() {
    _pageController.dispose();
    _emailCtrl.dispose();
    _phoneCtrl.dispose();
    _telegramCtrl.dispose();
    super.dispose();
  }

  void _goNext() {
    if (_page < 2) {
      _pageController.nextPage(
        duration: const Duration(milliseconds: 450),
        curve: Curves.easeInOutCubic,
      );
    }
  }

  Future<void> _saveAndContinue() async {
    if (!_formKey.currentState!.validate()) return;
    final email = _emailCtrl.text.trim();
    final phone = _phoneCtrl.text.trim();
    if (email.isEmpty && phone.isEmpty) {
      _showError('Vui lòng nhập ít nhất Email hoặc Số điện thoại.');
      return;
    }

    setState(() => _saving = true);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('email', email);
    await prefs.setString('phone', phone);
    await prefs.setString('telegram_chat_id', _telegramCtrl.text.trim());
    setState(() => _saving = false);
    _goNext();
  }

  Future<void> _checkServer() async {
    setState(() {
      _checkingServer = true;
      _serverOk = null;
    });
    final ok = await _api.checkHealth();
    setState(() {
      _checkingServer = false;
      _serverOk = ok;
    });
  }

  void _enter() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('onboarded', true);
    if (!mounted) return;
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => const HomeScreen()),
    );
  }

  void _showError(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(msg),
        backgroundColor: AppColors.danger,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: Stack(
        children: [
          // Background gradient blob
          Positioned(
            top: -120,
            left: -80,
            child: Container(
              width: 420,
              height: 420,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: RadialGradient(colors: [
                  AppColors.primary.withOpacity(.18),
                  Colors.transparent,
                ]),
              ),
            ),
          ),
          Positioned(
            bottom: -100,
            right: -60,
            child: Container(
              width: 320,
              height: 320,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: RadialGradient(colors: [
                  AppColors.danger.withOpacity(.12),
                  Colors.transparent,
                ]),
              ),
            ),
          ),

          // Page content
          PageView(
            controller: _pageController,
            physics: const NeverScrollableScrollPhysics(),
            onPageChanged: (p) => setState(() => _page = p),
            children: [
              _WelcomePage(onNext: _goNext),
              _ContactPage(
                formKey: _formKey,
                emailCtrl: _emailCtrl,
                phoneCtrl: _phoneCtrl,
                telegramCtrl: _telegramCtrl,
                saving: _saving,
                onSave: _saveAndContinue,
              ),
              _ServerPage(
                checking: _checkingServer,
                serverOk: _serverOk,
                onCheck: _checkServer,
                onEnter: _enter,
              ),
            ],
          ),

          // Dot indicators
          Positioned(
            bottom: 32,
            left: 0,
            right: 0,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: List.generate(3, (i) => _Dot(active: i == _page)),
            ),
          ),
        ],
      ),
    );
  }
}

// ── Page 1: Welcome ───────────────────────────────────────
class _WelcomePage extends StatelessWidget {
  final VoidCallback onNext;
  const _WelcomePage({required this.onNext});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 480),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Logo / icon
              Container(
                width: 100,
                height: 100,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(28),
                  gradient: const LinearGradient(
                    colors: [Color(0xFF0A84FF), Color(0xFF0055CC)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: AppColors.primary.withOpacity(.4),
                      blurRadius: 32,
                      offset: const Offset(0, 12),
                    ),
                  ],
                ),
                child: const Icon(Icons.shield_rounded,
                    size: 52, color: Colors.white),
              )
                  .animate()
                  .fadeIn(duration: 600.ms)
                  .scale(begin: const Offset(.8, .8)),
              const SizedBox(height: 32),
              Text('SafeWatch',
                      style: Theme.of(context)
                          .textTheme
                          .displayLarge
                          ?.copyWith(fontSize: 36))
                  .animate()
                  .fadeIn(delay: 200.ms, duration: 500.ms)
                  .slideY(begin: .2),
              const SizedBox(height: 12),
              Text(
                'Hệ thống phát hiện bạo lực học đường\nthông minh, cảnh báo tức thì',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                      height: 1.6,
                      color: AppColors.textSecondary,
                    ),
              )
                  .animate()
                  .fadeIn(delay: 350.ms, duration: 500.ms)
                  .slideY(begin: .2),
              const SizedBox(height: 48),
              _FeatureRow(
                icon: Icons.video_file_rounded,
                color: AppColors.primary,
                title: 'Phân tích video',
                sub: 'Upload video và phát hiện bạo lực tự động',
              ).animate().fadeIn(delay: 450.ms),
              const SizedBox(height: 16),
              _FeatureRow(
                icon: Icons.content_cut_rounded,
                color: AppColors.warning,
                title: 'Cắt clip chính xác',
                sub: 'Chỉ gửi đoạn có hành vi bạo lực, không cả video',
              ).animate().fadeIn(delay: 550.ms),
              const SizedBox(height: 16),
              _FeatureRow(
                icon: Icons.notifications_active_rounded,
                color: AppColors.success,
                title: 'Thông báo ngay lập tức',
                sub: 'Gửi cảnh báo qua Gmail và Telegram',
              ).animate().fadeIn(delay: 650.ms),
              const SizedBox(height: 48),
              ElevatedButton(
                onPressed: onNext,
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: const [
                    Text('Bắt đầu thiết lập'),
                    SizedBox(width: 8),
                    Icon(Icons.arrow_forward_rounded, size: 18),
                  ],
                ),
              ).animate().fadeIn(delay: 750.ms).slideY(begin: .3),
            ],
          ),
        ),
      ),
    );
  }
}

class _FeatureRow extends StatelessWidget {
  final IconData icon;
  final Color color;
  final String title, sub;
  const _FeatureRow(
      {required this.icon,
      required this.color,
      required this.title,
      required this.sub});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(
          width: 44,
          height: 44,
          decoration: BoxDecoration(
            color: color.withOpacity(.12),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Icon(icon, color: color, size: 22),
        ),
        const SizedBox(width: 16),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title,
                  style: Theme.of(context)
                      .textTheme
                      .titleMedium
                      ?.copyWith(fontSize: 14)),
              Text(sub, style: Theme.of(context).textTheme.bodyMedium),
            ],
          ),
        ),
      ],
    );
  }
}

// ── Page 2: Contact form ──────────────────────────────────
class _ContactPage extends StatelessWidget {
  final GlobalKey<FormState> formKey;
  final TextEditingController emailCtrl, phoneCtrl, telegramCtrl;
  final bool saving;
  final VoidCallback onSave;

  const _ContactPage({
    required this.formKey,
    required this.emailCtrl,
    required this.phoneCtrl,
    required this.telegramCtrl,
    required this.saving,
    required this.onSave,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 480),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 32),
          child: Form(
            key: formKey,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Thông tin liên hệ',
                    style: Theme.of(context).textTheme.displayMedium),
                const SizedBox(height: 8),
                Text(
                  'Nhập ít nhất Email hoặc Số điện thoại để nhận cảnh báo.',
                  style: Theme.of(context).textTheme.bodyLarge,
                ),
                const SizedBox(height: 32),

                // Email field
                _Label('Gmail (nhận email cảnh báo + clip)'),
                const SizedBox(height: 8),
                TextFormField(
                  controller: emailCtrl,
                  keyboardType: TextInputType.emailAddress,
                  decoration: const InputDecoration(
                    hintText: 'example@gmail.com',
                    prefixIcon: Icon(Icons.email_outlined, size: 20),
                  ),
                  validator: (v) {
                    if (v != null && v.isNotEmpty) {
                      if (!RegExp(r'^[\w.+-]+@[\w-]+\.\w+$').hasMatch(v)) {
                        return 'Email không hợp lệ';
                      }
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 20),

                // Phone field
                _Label('Số điện thoại (hiển thị trong thông báo)'),
                const SizedBox(height: 8),
                TextFormField(
                  controller: phoneCtrl,
                  keyboardType: TextInputType.phone,
                  decoration: const InputDecoration(
                    hintText: '0901234567',
                    prefixIcon: Icon(Icons.phone_outlined, size: 20),
                  ),
                ),
                const SizedBox(height: 20),

                // Telegram Chat ID
                _Label('Telegram Chat ID (nhận clip qua Telegram)'),
                const SizedBox(height: 8),
                TextFormField(
                  controller: telegramCtrl,
                  keyboardType: TextInputType.number,
                  decoration: const InputDecoration(
                    hintText: '123456789',
                    prefixIcon: Icon(Icons.telegram, size: 20),
                    helperText:
                        'Nhắn /start cho bot của bạn để lấy Chat ID',
                  ),
                ),

                const SizedBox(height: 12),
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: AppColors.primary.withOpacity(.08),
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(
                        color: AppColors.primary.withOpacity(.2)),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.info_outline,
                          color: AppColors.primary, size: 16),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          'Phải có ít nhất Email hoặc Telegram Chat ID để gửi cảnh báo.',
                          style:
                              Theme.of(context).textTheme.bodyMedium?.copyWith(
                                    color: AppColors.primary,
                                    fontSize: 12,
                                  ),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 32),
                ElevatedButton(
                  onPressed: saving ? null : onSave,
                  child: saving
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                              strokeWidth: 2, color: Colors.white),
                        )
                      : const Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text('Tiếp theo'),
                            SizedBox(width: 8),
                            Icon(Icons.arrow_forward_rounded, size: 18),
                          ],
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

class _Label extends StatelessWidget {
  final String text;
  const _Label(this.text);
  @override
  Widget build(BuildContext context) => Text(
        text,
        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              color: AppColors.textSecondary,
              fontWeight: FontWeight.w500,
            ),
      );
}

// ── Page 3: Server check ──────────────────────────────────
class _ServerPage extends StatelessWidget {
  final bool checking;
  final bool? serverOk;
  final VoidCallback onCheck, onEnter;

  const _ServerPage({
    required this.checking,
    required this.serverOk,
    required this.onCheck,
    required this.onEnter,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 480),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _ServerIcon(checking: checking, serverOk: serverOk),
              const SizedBox(height: 32),
              Text(
                checking
                    ? 'Đang kiểm tra kết nối...'
                    : serverOk == null
                        ? 'Kiểm tra kết nối Backend'
                        : serverOk!
                            ? 'Backend đang chạy! ✅'
                            : 'Không kết nối được Backend ❌',
                style: Theme.of(context).textTheme.displayMedium,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 12),
              Text(
                serverOk == null
                    ? 'Hãy đảm bảo backend đang chạy:\nuvicorn main:app --reload'
                    : serverOk!
                        ? 'SafeWatch sẵn sàng phân tích video bạo lực'
                        : 'Chạy: cd backend && uvicorn main:app --reload\nRồi kiểm tra lại',
                textAlign: TextAlign.center,
                style: Theme.of(context)
                    .textTheme
                    .bodyLarge
                    ?.copyWith(height: 1.6),
              ),
              const SizedBox(height: 40),
              if (!checking)
                ElevatedButton.icon(
                  onPressed: onCheck,
                  icon: const Icon(Icons.wifi_tethering_rounded),
                  label: const Text('Kiểm tra kết nối'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.surfaceAlt,
                    foregroundColor: AppColors.textPrimary,
                  ),
                ),
              if (serverOk == true) ...[
                const SizedBox(height: 16),
                ElevatedButton(
                  onPressed: onEnter,
                  child: const Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.rocket_launch_rounded),
                      SizedBox(width: 8),
                      Text('Vào ứng dụng'),
                    ],
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class _ServerIcon extends StatelessWidget {
  final bool checking;
  final bool? serverOk;
  const _ServerIcon({required this.checking, required this.serverOk});

  @override
  Widget build(BuildContext context) {
    if (checking) {
      return const SizedBox(
        width: 80,
        height: 80,
        child: CircularProgressIndicator(
          strokeWidth: 3,
          color: AppColors.primary,
        ),
      );
    }
    final color = serverOk == null
        ? AppColors.primary
        : serverOk!
            ? AppColors.success
            : AppColors.danger;
    final icon = serverOk == null
        ? Icons.dns_rounded
        : serverOk!
            ? Icons.check_circle_rounded
            : Icons.error_rounded;

    return Container(
      width: 80,
      height: 80,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: color.withOpacity(.12),
        border: Border.all(color: color.withOpacity(.3), width: 2),
      ),
      child: Icon(icon, size: 40, color: color),
    ).animate().scale(begin: const Offset(.8, .8));
  }
}

class _Dot extends StatelessWidget {
  final bool active;
  const _Dot({required this.active});

  @override
  Widget build(BuildContext context) => AnimatedContainer(
        duration: const Duration(milliseconds: 300),
        margin: const EdgeInsets.symmetric(horizontal: 4),
        width: active ? 20 : 6,
        height: 6,
        decoration: BoxDecoration(
          color: active ? AppColors.primary : AppColors.divider,
          borderRadius: BorderRadius.circular(3),
        ),
      );
}
