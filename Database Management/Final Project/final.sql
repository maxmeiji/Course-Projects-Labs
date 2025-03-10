-- phpMyAdmin SQL Dump
-- version 5.1.0
-- https://www.phpmyadmin.net/
--
-- 主機： 127.0.0.1
-- 產生時間： 2021-05-30 02:41:11
-- 伺服器版本： 10.4.18-MariaDB
-- PHP 版本： 7.4.16

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- 資料庫： `final`
--

-- --------------------------------------------------------

--
-- 資料表結構 `cart`
--

CREATE TABLE `cart` (
  `thing_id` int(6) NOT NULL,
  `name` varchar(50) NOT NULL,
  `price` int(11) NOT NULL,
  `amount` int(11) NOT NULL,
  `url` varchar(1000) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- 資料表結構 `commodity`
--

CREATE TABLE `commodity` (
  `c_id` int(6) NOT NULL,
  `c_name` varchar(50) CHARACTER SET utf8 COLLATE utf8_unicode_ci NOT NULL,
  `price` int(10) NOT NULL,
  `quantity` int(10) NOT NULL,
  `describe` varchar(1000) CHARACTER SET utf8 COLLATE utf8_unicode_ci NOT NULL,
  `url1` varchar(1000) CHARACTER SET utf8 COLLATE utf8_unicode_ci NOT NULL,
  `user_id` int(6) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- 傾印資料表的資料 `commodity`
--

INSERT INTO `commodity` (`c_id`, `c_name`, `price`, `quantity`, `describe`, `url1`, `user_id`) VALUES
(1, 'Cooler Master CH321 電競耳機麥克風', 990, 15, '★玩家首選 ★彈性的緩衝頭帶 ★清晰的語音通話 ★扎實的聲音 ★跨平台相容性 ★超高C/P值', 'https://d.ecimg.tw/items/DCAY0SA900ABPUV/000001_1571839133.jpg', 1),
(2, '森海塞爾 Sennheiser CX 150BT 無線藍牙頸掛入耳式耳機 - 黑色', 2490, 10, '■ 台灣公司貨 二年保固 ■ 高端無線藍牙5.0技術 ■ 10小時續航力 ■ 提供四種不同大小尺寸的耳墊 ■ 三鍵式操控界面 ■ 可同時連接兩台設備', 'https://b.ecimg.tw/items/DCAYMPA900AM5GL/000001_1605250588.jpg', 1),
(3, '森海塞爾 Sennheiser CX 400BT True Wireless 白色 真無線藍牙耳機', 6490, 20, '■ 台灣公司貨 二年保固 ■ 7mm超寬頻微型動圈單體 ■ 藍牙5.1高效能傳輸 ■ 降噪麥克風確保清晰收音', 'https://f.ecimg.tw/items/DCAY17A900B43FY/000001_1611289086.jpg', 1),
(4, '鐵三角 ATH-ANC500BT 無線抗噪耳機', 3500, 15, '■獨創主動式抗噪技術與藍牙無線技術 ■QuietPoint®主動式抗噪技術 ■藍牙無線技術，內建麥克風和控制鍵 ■Ø40mm驅動單元再現高傳真音效 ■1.2m導線可在電量不足時提供有線連接 ■可折疊設計使耳機更適合隨身攜帶 ■滿電後可使用藍牙與抗噪20小時 ■台灣鐵三角公司貨', 'https://d.ecimg.tw/items/DYAQ9DA900AWCYR/000001_1605597636.jpg', 2),
(5, '鐵三角 ATH-WS330BT 無線藍牙 耳罩式耳機', 3380, 20, '■立體縫製低反彈耳套 ■獨家設計的機殼(PAT.P) ■專用設計的Ø40mm SOLID BASS驅動單元 ■易攜帶的輕量小型耳罩式耳機 ■充電10分鐘即可使用約4小時 ■充飽電後最大可使用70小時 ■台灣鐵三角公司貨', 'https://e.ecimg.tw/items/DCAYKKA900A5FUJ/000001_1612248673.jpg', 2),
(6, 'acer Nitro 27吋2K HDR廣視角電競螢幕(XV272U V)', 9888, 13, '2560*1440 2K解析/平面/FreeSync/0.5ms/170Hz 2560x1440 2K高解析 IPS 178度廣視角 最高亮度：400 cd/㎡ ZERO FRAME 無邊框設計 AMD FreeSync技術 0.5ms急速反應時間 170Hz更新頻率 Display HDR 400認證 DCI-P3 95% 高色準Delta E2 Display Widget 公用程式軟體 內建喇叭2Wx2 支援HDMI 2.0x2/DisplayPort1.2介面 支援VESA壁掛', 'https://b.ecimg.tw/items/DSABHX1900BAODN/000001_1619605257.jpg', 2),
(7, '【ASUS TUF DASH F15】 ASUS FX516PM-0161C11370H 星耀白', 39900, 15, '處理器：Intel® Core™ i7-11370H Processor 3.3 GHz 記憶體：8GB DDR4 on board 硬碟：512GB M.2 NVMe™ PCIe® 3.0 SSD 獨立顯卡：NVIDIA® GeForce RTX™ 3060 6G獨顯 螢幕：15.6’(薄邊框)/FHD 1920x1080 16:9/144Hz /IPS-level 無線網路：\"Wi-Fi 6(802.11ax)+Bluetooth 5.1 (Dual band) 2*2 光碟機： 無 介面：HDMI、Thunderbolt™ 4 電池：76WHrs, 4S1P, 4-cell Li-ion 重量：2 KG 厚度：1.99cm 作業系統：Windows 10 (64bit) 保固：2年全球保固 + 首年免預付完美保固                ★全新顯卡視覺享受NVIDIA® GeForce RTX™ 3060獨顯6G效能 ★超屌效能 2021新世代 Intel® Core™ i7-11370H Processor 3.3 GHz ★超狂容量 512GB PCIe SSD ★全新升級 薄邊框螢幕 144Hz ★15吋FHD IPS 大屏佔比 ★通過軍規測試/防塵散熱通道 ★支援PD3.0充電技術', 'https://c.ecimg.tw/items/DHAS1UA900B8IH1/000001_1621307474.jpg', 3),
(8, 'ASUS ZenBook UX334FLC 13吋雙螢幕輕薄筆電', 30990, 12, 'LCD尺寸：13.3’//250nits//FHD 1920x1080 16:9//Anti-Glare//NTSC: 72%//Wide View 處理器：Intel® Core™ i5-10210U 1.6 GHz 記憶體：LDDR3 8G 2133 (Onboard 8G) 顯卡：NVIDIA MX 250 2G獨顯 硬碟：512GB M.2 NVMe™ PCIe® 3.0 SSD 其他：四邊NonaEdge極窄邊框、ScreenPad 2.0 網路：802.11ac+Bluetooth 5.0 (Dual band) 2*2 重量：1.19kg 作業系統： 64 Bits Windows 10 Home 保固：2年全球保固 LCD無亮點保固               16型IPS可攜式螢幕組合★遠距上課/居家上班首選!!!!!!!!', 'https://d.ecimg.tw/items/DHAXAQ1900BE5A7/000001_1621846084.jpg', 3),
(9, 'Cooler Master MasterCase H100 機殼(Mini-ITX專用)', 1490, 3, '★Mini-ITX專用 ★精細的通風網孔 ★標配200mm RGB風扇 ★相容ATX電源供應器 ★內建提把', 'https://d.ecimg.tw/items/DRAE1YA900A9MS8/000001_1569049490.jpg', 3),
(30, 'NS Switch 魔物獵人: 崛起 亞版中文版', 1790, 20, '以和風忍者村莊為舞台！ 為狩獵掀起全新風潮 新隨從「隨從加爾克」登場！ 可利用「翔蟲」縱橫馳騁', 'https://e.ecimg.tw/items/DGBJBH1900B9M4A/000001_1621995998.jpg', 1),
(31, '任天堂 Switch (電光藍/紅) 台灣公司貨遊戲主機 + 健身環大冒險', 11980, 10, 'NS新型號主機-電池持續時間加長!!! ▉ 最多連線8台，與親朋好友協力對戰 ▉ 多種操控模式及玩法，樂享家中大螢幕或外攜都適合 ▉ Joy-Con內置「HD震動」體驗逼真細膩臨場感 ◆ 支援Nintendo Switch Online (※付費，台灣地區暫未支援) ◆ 支援繁體中文介面 ◆ 台灣公司貨，提供1年保固。', 'https://f.ecimg.tw/items/DGBJDE1900B535Z/000001_1619575675.jpg', 1),
(32, 'NS Switch遊戲 太鼓之達人 咚咔！ 二合一大冒險-中文版', 1390, 15, '太鼓達人RPG！？演奏太鼓來拯救世界！ ▉《時空大冒險》與《神秘冒險記》兩款重製合輯 ▉融合RPG玩法，操作小咚小喀穿越時空冒險去！ ▉收錄「紅蓮華」等6首新曲共130首 ▉登場怪物超過250種，用演奏一較高下 ▉打敗怪物讓他們成為夥伴，組成專屬隊伍 ▉ 您參考的商品為保護級', 'https://f.ecimg.tw/items/DGBJCWA900AVNYG/000001_1608171394.jpg', 1),
(33, 'Switch《瑪利歐賽車8 豪華版》', 1790, 12, '▉ 您目前所參考的商品為保護級 ▉ 收錄所有『瑪利歐賽車 8』以及追加下載內容 ▉ 還有新角色、對戰賽道以及車輛 ▉ 擴充為 5 種對戰規則 ▉ 適合全家一起玩的競速遊戲 ▉ 支援 繁體中文', 'https://f.ecimg.tw/items/DGBJBH1900BC4RR/000001_1620617836.jpg', 1),
(34, 'NS Switch遊戲 寶可夢 劍-中文版', 1790, 10, '「極巨化」寶可夢開戰！全新舞台展開冒險 ▉ 全新3隻寶可夢，攜手前進吧！ ▉ 傳說的「蒼響」與「藏瑪然特」究竟是？ ▉ 施展超強威力～「極巨招式」！ ▉ 4人協力對抗極巨化野生寶可夢 ▉ 體驗英國場景 千變萬化「曠野地帶」 ▉ 您參考的商品為保護級', 'https://e.ecimg.tw/items/DGBJCWA900A34NA/000001_1613533367.jpg', 1),
(35, 'NS Switch遊戲 路易吉洋樓3 (路易吉鬼屋3)-中文版', 1690, 10, '廢棄鬼屋洋樓 掃蕩幽靈解救伙伴！ ▉ 追加眾多要素，抓鬼更刺激！更有趣 ▉ 膽小鬼路易吉擁有抓鬼新能力！？ ▉「果凍吉」可穿越障礙物協助路易吉 ▉ 結合驚魂抓鬼x冒險解謎的眾多樂趣 ▉「驚魂塔」最多支援8人同樂 ▉ 您參考的商品為保護級', 'https://c.ecimg.tw/items/DGBJCWA900A75XL/000001_1613553784.jpg', 1),
(36, 'Switch《超級瑪利歐派對》支援中文', 1790, 10, '▉ 您目前所參考的商品為保護級 ▉ 超級派對遊戲登場 ▉ 透過新型擲骰子形式，戰略性變得更豐富 ▉ 增加了目前為止從未享受過的各種玩法 ▉ 隨時隨地都可和任何人一起遊玩的派對遊戲 ▉ 收錄80種小遊戲 ▉ 遊戲人數：1～4人 ▉ 支援繁體中文', 'https://f.ecimg.tw/items/DGBJBH1900BC4RV/000001_1620617895.jpg', 1),
(37, 'Switch遊戲 薩爾達傳說 曠野之息-中文版(台灣公司貨)', 1980, 10, '▉ 您參考的商品為保護級 ▉ 首度細膩打造廣大開放的3D海拉魯世界 ▉ 擬真物理模擬設計 更真實的“與自然環境產生的高度互動” ▉「Open Air」概念呼應宏偉世界的細膩BGM ▉ 豐富精心的謎題挑戰，巧思設計讓人耳目一新 ▉ 每秒鐘都屏氣凝神，無法錯過的興奮戰鬥', 'https://b.ecimg.tw/items/DGBJDJA9008RBSH/000001_1622003423.jpg', 1),
(38, '賽先生科學工廠｜克萊因瓶Klein Bottle-迷你款9.5cm', 791, 5, '● 跳脫物體的邊界，裡外穿梭自如 ● 1882年由數學家Felix Klein所推論 ● 款式有迷你款9.5cm和大款15.5cm', 'https://c.ecimg.tw/items/DEAE7EA90079C8I/000001_1622021048.jpg', 3),
(39, '【chicco】Goody Plus魔術瞬收手推車-酷灰', 8990, 5, '●單手魔術瞬收瞬開，新手爸媽必備 ●站立便攜，輕巧隨行，可登機 ●可調座椅，平躺舒適', 'https://d.ecimg.tw/items/DEAU5YA900B4U1W/000001_1621821694.jpg', 3),
(40, '【布布童鞋】玩具總動員抱抱龍銀色兒童電燈涼鞋[ B9K089Q ]', 789, 5, '震動感應式防水LED燈 雙魔鬼氈可依腳型調整 台灣製造，正版卡通授權', 'https://d.ecimg.tw/items/QFARDCA9009ZWQG/000001_1557304919.jpg', 3),
(41, '【mamaway 媽媽餵】迪士尼系列(玩具總動員)蠶寶寶包巾組(2入)', 1580, 5, '● 精梳棉材質，柔軟親膚 ●操作簡單，新手爸媽更容易上手 ●細緻柔軟魔鬼氈，不刮傷寶寶 ●彈性佳，新生兒寶寶身高/體型不受限 ●可依寶寶的大小調整緊度 ● 一包2入組合，方便清洗替換 ●可愛迪士尼系列圖案，三款花色可選  【適用年齡 / Age】 新生兒 (3.17~6.35公斤)', 'https://c.ecimg.tw/items/QFAQCTA900ATJ9J/000001_1597399114.jpg', 3),
(42, '賽先生科學工廠｜Plasma 電漿球/靜電球(USB hub功能)', 607, 10, '●彷彿是有條閃電跟著手移動 ●超魔幻視覺享受 ●4個USB擴充接口可當做USB集線器使用 ●電源可開關，美觀兼具實用 ●具備熱插拔功能，不易損壞設備', 'https://b.ecimg.tw/items/DEAE7EA90079BUV/000001_1622006474.jpg', 3),
(43, '賽先生科學工廠｜牛頓球 / 慣性原理擺動球', 519, 10, '● 搖擺球是一種相當歐洲相當古老的擺飾玩具 ● 療鬱小物 辦公桌上的擺飾 ● 碰撞聲音大聲而清脆', 'https://d.ecimg.tw/items/DEAE7EA90079B4T/000001_1619511077.jpg', 3),
(44, '【Berndes德國寶迪】德國製平衡系列不沾深炒鍋28cm(含鍋蓋)(IH爐適用)', 1980, 80, '＊德國原裝進口 ＊減少油量使用，少油煙,達到健康烹調飲食 ＊高強度的防刮傷設計，非常適合初學者使用。 ＊其鍋底使用拋光電磁網格狀的專利設計，不挑爐具，電磁爐ＩＨ爐皆可使用。 ＊不易沾黏,好清洗 ＊不含全氟辛酸氨,安全無虞 ＊輕巧耐用', 'https://f.ecimg.tw/items/DEAWMWA900ATXTJ/000001_1621822931.jpg', 2),
(45, '【BEKA貝卡】Pro Induc茵杜克晶鑽 雙耳附蓋湯鍋24cm(5113071244)', 2000, 80, '鍋身採用鋁合金鑄造 內層採用貝卡耐不沾Essential塗層 符合人體工學設計電木把手 不含PFOA，無毒又環保', 'https://d.ecimg.tw/items/DEAWDFA9007HTJJ/000001_1592361129.jpg', 2);

-- --------------------------------------------------------

--
-- 資料表結構 `user`
--

CREATE TABLE `user` (
  `id` int(6) UNSIGNED NOT NULL,
  `first_name` varchar(20) COLLATE utf8_unicode_ci NOT NULL,
  `last_name` varchar(20) COLLATE utf8_unicode_ci NOT NULL,
  `username` varchar(30) COLLATE utf8_unicode_ci NOT NULL,
  `password` varchar(1000) COLLATE utf8_unicode_ci NOT NULL,
  `card` char(16) COLLATE utf8_unicode_ci NOT NULL,
  `address` text COLLATE utf8_unicode_ci NOT NULL,
  `phone_num` char(30) COLLATE utf8_unicode_ci NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- 傾印資料表的資料 `user`
--

INSERT INTO `user` (`id`, `first_name`, `last_name`, `username`, `password`, `card`, `address`, `phone_num`) VALUES
(1, 'er', 'z', '宇生', '4a7d1ed414474e4033ac29ccb8653d9b', '12', '', '0988-670-444'),
(2, 'Kevin', 'Durant', 'KD', '1c383cd30b7c298ab50293adfecb7b18', '09999999', '', '0911111111'),
(3, '曉明', '王', '禁藥王', '4a7d1ed414474e4033ac29ccb8653d9b', '098888888', '', '0947568241'),
(4, 'er', 'z', '1', 'c4ca4238a0b923820dcc509a6f75849b', '1', '', '1'),
(5, 'er', 'z', '123', 'c20ad4d76fe97759aa27a0c99bff6710', '123', '123', '123'),
(6, 'er', 'z', '234', '289dff07669d7a23de0ef88d2f7129e7', '234', '234', '234'),
(7, 'er', 'z', 'm', '6f8f57715090da2632453988d9a1501b', 'm', '234', '234'),
(8, 'm', 'm', 'o', 'd95679752134a2d9eb61dbd7b91c4bcc', 'o', 'o', 'o');

--
-- 已傾印資料表的索引
--

--
-- 資料表索引 `commodity`
--
ALTER TABLE `commodity`
  ADD PRIMARY KEY (`c_id`);

--
-- 資料表索引 `user`
--
ALTER TABLE `user`
  ADD PRIMARY KEY (`id`);

--
-- 在傾印的資料表使用自動遞增(AUTO_INCREMENT)
--

--
-- 使用資料表自動遞增(AUTO_INCREMENT) `commodity`
--
ALTER TABLE `commodity`
  MODIFY `c_id` int(6) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=59;

--
-- 使用資料表自動遞增(AUTO_INCREMENT) `user`
--
ALTER TABLE `user`
  MODIFY `id` int(6) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=9;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
