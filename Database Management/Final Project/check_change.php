<?php

require_once 'db.php';
@session_start();
function check($old_pw, $new_pw)
{
	
  
  $result = null;
 
  //先把密碼用md5加密
  $old_pw = md5($old_pw);
  $new_pw = md5($new_pw);
  //將查詢語法當成字串，記錄在$sql變數中
  if($old_pw==$_SESSION['password'])
  {
	$sql = "UPDATE `user` SET `password` = '{$new_pw}' WHERE `username` = '{$_SESSION['username']}';";

  
  $query = mysqli_query($_SESSION['link'], $sql);


  //如果請求成功
  if ($query)
  {
    
    if(mysqli_affected_rows($_SESSION['link']) >0)
    {

      $result = true;
    }
	else{
		$result = false;
		
	}
  }
  else
  {

  }		
  }
  
  else{
	  
	  echo "兩次密碼輸入不一樣，請確認";
	  $result = false;
	  
  }
  

  //回傳結果
  return $result;
}





if(check($_POST['op'], $_POST['np']))
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