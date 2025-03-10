<?php

require_once 'db.php';
@session_start();
function verify_user($username, $password)
{	
	
  $result = null;
  //先把密碼用md5加密
  $password = md5($password);
  
  $sql = "SELECT * FROM `user` WHERE `username` = '{$username}' AND `password` = '{$password}'";

  
  $query = mysqli_query($_SESSION['link'], $sql);

  //如果請求成功
  if ($query)
  {
    
    if(mysqli_num_rows($query) == 1)
    {

      //取得使用者資料
      $user = mysqli_fetch_assoc($query);
      
     
	//setcookie("is_login",true);
	//setcookie("username",$user['username']);
	//setcookie("password",$user['password']);
	//setcookie("ident",$user['ident']);

	//setcookie("reg_date",$user['reg_date']);
	//setcookie("gender",$user['gender']);
	
	
	//setcookie("LAST_ACTIVITY",$_SERVER['REQUEST_TIME']);
	
	$_SESSION['is_login'] = TRUE;
    $_SESSION['username'] = $user['username'];
	$_SESSION['password'] = $user['password'];
	$_SESSION['card'] = $user['card'];
	$_SESSION['phone'] = $user['phone_num'];
	$_SESSION['lname'] = $user['last_name'];
	$_SESSION['LAST_ACTIVITY'] =$_SERVER['REQUEST_TIME'];
	$_SESSION['add'] = $user['address'];
	$_SESSION['id']=$user['id'];
	
	

      $result = true;
    }

  }
  else
  {
    echo "{$sql} 語法執行失敗，錯誤訊息：" . mysqli_error($_SESSION['link']);
  }

  //回傳結果
  return $result;
}






if(verify_user($_POST['un'], $_POST['pw']))
{
	//若為true 代表登入成功，印出yes
	echo 'yes';
}
else
{
	//若為 null 或者 false 代表失敗
	echo 'no';	
}

?>