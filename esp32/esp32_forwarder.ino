/*
  EntroGuard ESP32 — Network Traffic Forwarder
  =============================================
  Connects to mobile hotspot (2.4GHz only),
  captures its own traffic metadata, and sends
  feature vectors to Raspberry Pi via UDP.

  Board    : ESP32 Dev Module
  IDE      : Arduino IDE 2.x
  Library  : ArduinoJson (install via Library Manager)

  ─────────────────────────────────────────────
  SETUP INSTRUCTIONS:
  1. Set HOTSPOT_SSID and HOTSPOT_PASSWORD below
  2. Set PI_IP to your Pi's IP on the hotspot
     (find it by running: hostname -I on the Pi)
  3. Install ArduinoJson via Library Manager
  4. Flash to ESP32

  MOBILE HOTSPOT SETUP (IMPORTANT):
  • iPhone : Settings → Personal Hotspot
             → Enable "Maximize Compatibility"
  • Android: Settings → Hotspot → AP Band → 2.4GHz

  ESP32 does NOT support 5GHz WiFi!
  ─────────────────────────────────────────────
*/

#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>

// ─────────────────────────────────────────────
// ⚙️  CONFIG — Edit these before flashing
// ─────────────────────────────────────────────

const char* HOTSPOT_SSID     = "YOUR_HOTSPOT_NAME";      // ← your phone hotspot name
const char* HOTSPOT_PASSWORD = "YOUR_HOTSPOT_PASSWORD";  // ← your hotspot password
const char* PI_IP            = "192.168.XXX.XXX";        // ← Pi's IP on hotspot
                                                          //   run: hostname -I on Pi
const int   PI_PORT          = 5005;   // UDP port Pi listens on (don't change)
const int   WINDOW_SEC       = 5;      // feature reporting interval in seconds

// ─────────────────────────────────────────────
// GLOBALS
// ─────────────────────────────────────────────

WiFiUDP udp;

volatile uint32_t pkt_count   = 0;
volatile uint32_t syn_count   = 0;
volatile uint32_t ack_count   = 0;
volatile uint32_t rst_count   = 0;
volatile uint32_t total_bytes = 0;

float    last_pkt_time = 0;
float    iat_sum       = 0;
float    iat_sq_sum    = 0;
uint32_t iat_count     = 0;

uint8_t  port_seen[128] = {0};  // bitmap for ports 0–1023
uint16_t unique_ports   = 0;

unsigned long window_start = 0;
uint32_t      window_num   = 0;

// ─────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────

void markPort(uint16_t port) {
  if (port < 1024) {
    uint16_t byte_idx = port / 8;
    uint8_t  bit_idx  = port % 8;
    if (!(port_seen[byte_idx] & (1 << bit_idx))) {
      port_seen[byte_idx] |= (1 << bit_idx);
      unique_ports++;
    }
  }
}

void resetWindow() {
  pkt_count    = 0;
  syn_count    = 0;
  ack_count    = 0;
  rst_count    = 0;
  total_bytes  = 0;
  iat_sum      = 0;
  iat_sq_sum   = 0;
  iat_count    = 0;
  unique_ports = 0;
  memset(port_seen, 0, sizeof(port_seen));
  last_pkt_time = millis() / 1000.0;
  window_start  = millis();
}

float approxPortEntropy() {
  if (pkt_count == 0 || unique_ports == 0) return 0.0;
  float ratio = (float)unique_ports / (float)(pkt_count < 100 ? pkt_count : 100);
  if (ratio > 1.0) ratio = 1.0;
  return ratio * 4.0;
}

float iatVariance() {
  if (iat_count < 2) return 0.0;
  float mean = iat_sum / iat_count;
  float var  = (iat_sq_sum / iat_count) - (mean * mean);
  return var < 0 ? 0 : var;
}

void recordPacket(uint16_t dst_port, uint16_t size,
                  bool isSYN, bool isACK, bool isRST) {
  float now = millis() / 1000.0;

  pkt_count++;
  total_bytes += size;

  if (isSYN) syn_count++;
  if (isACK) ack_count++;
  if (isRST) rst_count++;

  markPort(dst_port);

  if (last_pkt_time > 0) {
    float iat   = now - last_pkt_time;
    iat_sum    += iat;
    iat_sq_sum += iat * iat;
    iat_count++;
  }
  last_pkt_time = now;
}

// ─────────────────────────────────────────────
// SEND FEATURES TO RASPBERRY PI VIA UDP
// ─────────────────────────────────────────────

void sendFeaturesToPi() {
  // Check WiFi still connected
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("⚠ WiFi disconnected — skipping window");
    return;
  }

  float elapsed     = (millis() - window_start) / 1000.0;
  if (elapsed <= 0)   elapsed = WINDOW_SEC;

  float packet_rate = pkt_count / elapsed;
  float syn_ratio   = pkt_count > 0 ? (float)syn_count / pkt_count : 0;
  float ack_ratio   = pkt_count > 0 ? (float)ack_count / pkt_count : 0;
  float rst_ratio   = pkt_count > 0 ? (float)rst_count / pkt_count : 0;
  float avg_size    = pkt_count > 0 ? (float)total_bytes / pkt_count : 0;
  float iat_var     = iatVariance();
  float port_ent    = approxPortEntropy();

  // Build JSON
  StaticJsonDocument<512> doc;
  doc["window"]            = window_num;
  doc["packet_rate"]       = packet_rate;
  doc["syn_ratio"]         = syn_ratio;
  doc["ack_ratio"]         = ack_ratio;
  doc["rst_ratio"]         = rst_ratio;
  doc["src_ip_entropy"]    = 0.0;        // ESP32 = single source always
  doc["dst_port_entropy"]  = port_ent;
  doc["unique_src_count"]  = 1;
  doc["unique_port_count"] = unique_ports;
  doc["avg_packet_size"]   = avg_size;
  doc["iat_variance"]      = iat_var;
  doc["device_ip"]         = WiFi.localIP().toString();
  doc["rssi"]              = WiFi.RSSI();

  char payload[512];
  serializeJson(doc, payload);

  // Send UDP packet to Pi
  udp.beginPacket(PI_IP, PI_PORT);
  udp.print(payload);
  udp.endPacket();

  // Serial monitor output
  Serial.println("\n========================================");
  Serial.printf("  Window #%d\n", window_num);
  Serial.printf("  ESP32 IP    : %s\n", WiFi.localIP().toString().c_str());
  Serial.printf("  Pi Target   : %s:%d\n", PI_IP, PI_PORT);
  Serial.printf("  RSSI        : %d dBm\n", WiFi.RSSI());
  Serial.println("  ----------------------------------------");
  Serial.printf("  Packets     : %d  (%.2f pkt/s)\n", pkt_count, packet_rate);
  Serial.printf("  SYN Ratio   : %.4f\n", syn_ratio);
  Serial.printf("  ACK Ratio   : %.4f\n", ack_ratio);
  Serial.printf("  RST Ratio   : %.4f\n", rst_ratio);
  Serial.printf("  Port Entropy: %.4f\n", port_ent);
  Serial.printf("  Unique Ports: %d\n", unique_ports);
  Serial.printf("  Avg Pkt Size: %.2f bytes\n", avg_size);
  Serial.printf("  IAT Variance: %.6f\n", iat_var);
  Serial.println("  ✅ Sent to Pi");
  Serial.println("========================================");
}

// ─────────────────────────────────────────────
// SIMULATE TRAFFIC
// Replace this with real recordPacket() calls
// in your actual application send/recv handlers
// ─────────────────────────────────────────────

void simulateTrafficActivity() {
  uint16_t common_ports[] = {80, 443, 8080, 53, 22, 3000};
  int packets_this_tick   = random(5, 25);

  for (int i = 0; i < packets_this_tick; i++) {
    uint16_t port = common_ports[random(0, 6)];
    uint16_t size = random(64, 1400);
    bool isSYN    = (random(0, 10) < 2);
    bool isACK    = (random(0, 10) < 7);
    bool isRST    = (random(0, 10) < 1);
    recordPacket(port, size, isSYN, isACK, isRST);
    delay(random(10, 50));
  }
}

// ─────────────────────────────────────────────
// WIFI CONNECTION WITH RETRY
// ─────────────────────────────────────────────

void connectToHotspot() {
  Serial.printf("\nConnecting to hotspot: %s\n", HOTSPOT_SSID);
  Serial.println("(Make sure hotspot is set to 2.4GHz!)");

  WiFi.mode(WIFI_STA);
  WiFi.begin(HOTSPOT_SSID, HOTSPOT_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    attempts++;
    if (attempts > 40) {
      Serial.println("\n❌ Failed to connect. Check:");
      Serial.println("   - Hotspot name and password in code");
      Serial.println("   - Hotspot is set to 2.4GHz");
      Serial.println("   - Hotspot is turned on");
      Serial.println("   Restarting in 5s...");
      delay(5000);
      ESP.restart();
    }
  }

  Serial.println("\n✅ Connected to hotspot!");
  Serial.printf("   ESP32 IP : %s\n", WiFi.localIP().toString().c_str());
  Serial.printf("   Gateway  : %s\n", WiFi.gatewayIP().toString().c_str());
  Serial.printf("   Signal   : %d dBm\n", WiFi.RSSI());
}

// ─────────────────────────────────────────────
// SETUP
// ─────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n========================================");
  Serial.println("  EntroGuard ESP32 — Traffic Forwarder");
  Serial.println("  Mobile Hotspot Edition (2.4GHz)");
  Serial.println("========================================");

  connectToHotspot();

  // Start UDP
  udp.begin(5006);
  Serial.printf("\n📡 UDP ready — sending to %s:%d\n", PI_IP, PI_PORT);
  Serial.printf("   Window interval: %d seconds\n\n", WINDOW_SEC);

  resetWindow();
  randomSeed(analogRead(0));
}

// ─────────────────────────────────────────────
// LOOP
// ─────────────────────────────────────────────

void loop() {
  // Auto-reconnect if hotspot drops
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("⚠ Hotspot connection lost. Reconnecting...");
    connectToHotspot();
    resetWindow();
    return;
  }

  // Record traffic activity
  simulateTrafficActivity();

  // Send features to Pi every WINDOW_SEC
  if (millis() - window_start >= (unsigned long)(WINDOW_SEC * 1000)) {
    window_num++;
    sendFeaturesToPi();
    resetWindow();
  }
}
