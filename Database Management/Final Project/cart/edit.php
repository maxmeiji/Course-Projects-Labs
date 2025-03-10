<?php
session_start();
$UserId=$_SESSION['id'];
?>
<header>
<h1 style= "background-image: url(https://img.tukuppt.com/ad_preview/00/12/30/5c995b18c5cf6.jpg!/fw/780);height: 180px; line-height:80px;font-family:微軟正黑體;text-align: center;color: white"  >  YAPPI SHOP 貨架資料庫 <br>精於心，簡於形，滿足您的心
			
</h1>	
</header>
<center><marquee  bgcolor=#3C3C3C width="1480" behavior=side direction="left" onMouseOver="this.stop()" onMouseOut="this.start()"  
 ><font color="	#FFFFFF" size=5 family=華康行書體 >YAPPI 公告 : 請各位賣家將欲售商品上傳於此貨架，方便YAPPI為您處理商品相關事宜，進行交易前請先詳閱消費相關權益條款，祝福您交易愉快!    *防疫期間，也請大家戴口罩，勤洗手，台灣加油!大家加油!*</font></marquee> </center>
<body bgcolor="#cccccc">
<table "style=border:3px #cccccc solid;" cellpadding="10" border="1" align="center" valign="center">
<tr><th bgcolor=#3C3C3C ><font color="white">商品<br>編號</font></th><th bgcolor=#3C3C3C ><font color="white">---商品名稱---</font></th><th bgcolor=#3C3C3C ><font color="white">---價格---</font></th><th bgcolor=#3C3C3C ><font color="white">---庫存---</font></th><th bgcolor=#3C3C3C ><font color="white">---商品描述---</font></th><th bgcolor=#3C3C3C ><font color="white">---商品圖片連結---</font></th><th colspan="2" bgcolor=#3C3C3C ><font color="white">---資料修改確認---</font></th></tr>
<?php
$pdo=new PDO('mysql:host=localhost;dbname=final;charset=utf8', 'root', '');
if (isset($_REQUEST['command'])) {
	switch ($_REQUEST['command']) {
	case 'insert':
		if (empty($_REQUEST['c_name']) || 
			!preg_match('/[0-9]+/', $_REQUEST['price'])) break;
		$sql=$pdo->prepare('insert into commodity values(null,?,?,?,?,?,?)');
		$sql->execute(
			[htmlspecialchars($_REQUEST['c_name']), $_REQUEST['price'],$_REQUEST['quantity'],$_REQUEST['describe'],$_REQUEST['url1'], $UserId]);
		break;
	case 'update':
		if (empty($_REQUEST['c_name']) || !preg_match('/[0-9]+/', $_REQUEST['price'])||!preg_match('/[0-9]+/', $_REQUEST['quantity'])) break;
		$sql=$pdo->prepare('UPDATE commodity SET c_name=?, price=?, quantity=? WHERE c_id=? AND user_id=?');
		$sql->execute([$_REQUEST['c_name'], $_REQUEST['price'],$_REQUEST['quantity'], $_REQUEST['id'], $UserId]);
		break;
	case 'delete':
		$sql=$pdo->prepare('DELETE FROM commodity WHERE c_id=?');
		$sql->execute([$_REQUEST['c_id']]);
		break;
	}
}
foreach ($pdo->query("select * from commodity where user_id=$UserId") as $row) {
	echo '<tr >';
	echo '<form action="edit.php" method="post">';
	echo '<input type="hidden" name="command" value="update">';
	echo '<input type="hidden" name="id" value="', $row['c_id'], '">';
	echo '<td bgcolor=#3C3C3C style="color:white;" >', $row['c_id'], '</td>';

	echo '<td>';
	echo '<input type="text" name="c_name" size="60" align="center" valign="center" value="', $row['c_name'], '">';
	echo '</td>';
	echo '<td>';
	echo '<input type="text" name="price" size="10" align="center" valign="center" value="', $row['price'], '">';
	echo '</td>';
    echo '<td>';
	echo '<input type="text" name="quantity" size="10" align="center" valign="center" value="', $row['quantity'], '">';
	echo '</td>';
    echo '<td>';
	echo '<input type="text" name="describe" size="35" align="center" valign="center" value="', $row['describe'], '">';
	echo '</td>';
    echo '<td>';
	echo '<input type="text" name="url1" size="30" align="center" valign="center" value="', $row['url1'], '">';
	echo '</td>';
	echo '<td><input type="submit" value="確定修改"></td>';
	echo '</form>';
	echo '<form action="edit.php" method="post">';
	echo '<input type="hidden" name="command" value="delete">';
	echo '<input type="hidden" name="c_id" value="', $row['c_id'], '">';
	echo '<td><input type="submit" value="確定刪除"></td>';
	echo '</form>';
	echo '</tr>';
	echo "\n";
}
?>
<tr>
<form action="edit.php" method="post">
<input type="hidden" name="command" value="insert">
<td></td>
<td><input type="text" name="c_name" size="60" align="center" valign="center"></td>
<td><input type="text" name="price"  size="10" align="center" valign="center"></td>
<td><input type="text" name="quantity" size="10" align="center" valign="center"></td>
<td><input type="text" name="describe" size="35" align="center" valign="center"></td>
<td><input type="text" name="url1" size="30" align="center" valign="center"></td>
<td align="center" colspan="2"><input type="submit" value="新增品項" size="35" align="center"></td>
</form>
</tr>
</table>
<div class=" "align="center"><a class="btn btn-primary" href="../view/final2.php" >回到首頁</a></div> <!--從cart(shelf位置)跳出來，電腦看得懂:"../"->view->final2(這個是HTML的註解方式)-->
