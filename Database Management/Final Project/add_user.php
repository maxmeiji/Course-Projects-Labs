<?php

require_once 'db.php';
@session_start();

function add_user( $lname, $fname, $username, $password, $card, $phone, $add )
{	

	
  $result = null;
	//先把密碼用md5加密
	$password = md5($password);
  
  $sql = "INSERT INTO `user` (`last_name`, `first_name`, `username` ,`password`, `card`, `phone_num`, `address`) VALUE ('{$lname}', '{$fname}', '{$username}','{$password}', '{$card}', '{$phone}', '{$add}');";


  $query = mysqli_query($_SESSION['link'], $sql);

 
  if ($query)
  {
    
    if(mysqli_affected_rows($_SESSION['link']) == 1)
    {
     
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


$add_result = add_user($_POST['ln'], $_POST['fn'], $_POST['un'], $_POST['pw'], $_POST['cd'], $_POST['p'], $_POST['ad']);

if($add_result)
{
	//若為true 代表新增成功，印出yes
	echo 'yes';
}
else
{
	//若為 null 或者 false 代表失敗
	echo 'no';	
}

?>
